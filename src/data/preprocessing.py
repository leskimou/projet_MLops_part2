# Ce fichier constitue le pipeline complet de prétraitement des données pour
# un modèle de scoring de crédit (type Home Credit Default Risk).
#
# L'objectif est de fusionner plusieurs sources de données (demandes de prêt,
# historique bureau de crédit, remboursements, cartes de crédit, etc.) en un
# seul DataFrame prêt à l'entraînement d'un modèle de machine learning.
#
# Flux général :
#   application_train.csv          → données principales du demandeur
#   bureau.csv + bureau_balance.csv → historique crédits externes (Credit Bureau)
#   previous_application.csv        → demandes de prêt passées chez Home Credit
#   POS_CASH_balance.csv            → soldes mensuels des prêts/POS précédents
#   installments_payments.csv       → historique des paiements d'échéances
#   credit_card_balance.csv         → soldes mensuels des cartes de crédit
#
# Toutes ces tables sont agrégées par client (SK_ID_CURR) puis fusionnées.
# =============================================================================

import os           
import gc          
import time         
import re           
import numpy as np
import pandas as pd
from contextlib import contextmanager   
import multiprocessing as mp          
from functools import partial          
from scipy.stats import kurtosis, iqr, skew
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)



# =============================================================================
# FONCTION PRINCIPALE : load_dataset
# =============================================================================

def load_dataset(debug=False):
    """
    Charge et prétraite toutes les sources de données, puis retourne le
    DataFrame final fusionné (données d'entraînement uniquement).

    Paramètre :
        debug (bool) : Si True, charge seulement 30 000 lignes par fichier
                       pour accélérer les tests. Par défaut False (toutes les données).

    Retourne :
        df (pd.DataFrame) : DataFrame final avec toutes les features engineered,
                            prêt pour l'entraînement d'un modèle ML.
    """
    # En mode debug, on limite à 30 000 lignes pour aller plus vite
    num_rows = 30000 if debug else None

    #  Chargement des données principales des demandeurs de prêt 
    with timer("application_train"):
        df = get_train(DATA_DIRECTORY, num_rows=num_rows)
        print("Application dataframe shape: ", df.shape)

    #  Fusion avec les données du bureau de crédit externe 
    # Le bureau de crédit contient l'historique des crédits ouverts ailleurs
    with timer("Bureau and bureau_balance data"):
        bureau_df = get_bureau(DATA_DIRECTORY, num_rows=num_rows)
        df = pd.merge(df, bureau_df, on='SK_ID_CURR', how='left')
        print("Bureau dataframe shape: ", bureau_df.shape)
        del bureau_df; gc.collect()  # Libère la mémoire immédiatement après fusion

    #  Fusion avec les demandes de prêt précédentes chez Home Credit 
    with timer("previous_application"):
        prev_df = get_previous_applications(DATA_DIRECTORY, num_rows)
        df = pd.merge(df, prev_df, on='SK_ID_CURR', how='left')
        print("Previous dataframe shape: ", prev_df.shape)
        del prev_df; gc.collect()

    #  Fusion avec les balances mensuelles des prêts/POS, remboursements et cartes 
    with timer("previous applications balances"):
        # Soldes mensuels des prêts à la consommation / POS (Point Of Sale)
        pos = get_pos_cash(DATA_DIRECTORY, num_rows)
        df = pd.merge(df, pos, on='SK_ID_CURR', how='left')
        print("Pos-cash dataframe shape: ", pos.shape)
        del pos; gc.collect()

        # Historique détaillé des paiements d'échéances
        ins = get_installment_payments(DATA_DIRECTORY, num_rows)
        df = pd.merge(df, ins, on='SK_ID_CURR', how='left')
        print("Installments dataframe shape: ", ins.shape)
        del ins; gc.collect()

        # Soldes mensuels des cartes de crédit
        cc = get_credit_card(DATA_DIRECTORY, num_rows)
        df = pd.merge(df, cc, on='SK_ID_CURR', how='left')
        print("Credit card dataframe shape: ", cc.shape)
        del cc; gc.collect()

    #  Post-traitement final 
    df = add_ratios_features(df)        # Ajout de ratios croisant plusieurs sources
    df = replace_infinite_with_nan(df)  # Remplace les +/-inf par NaN (divisions par zéro)
    df = reduce_memory(df)              # Réduit l'empreinte mémoire en castant les types numériques
    df = sanitize_feature_names(df)     # Nettoie les noms de colonnes (caractères spéciaux, doublons)

    # Suppression colonnes > 40 % NaN
    nan_pct = df.isna().mean() * 100
    df = df.drop(columns=nan_pct[nan_pct > 40].index.tolist())

    # Suppression quasi-constantes (>= 99 % même valeur)
    PROTECTED_COLS = ["SK_ID_CURR", "TARGET"]
    dominant_ratio = df.apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
    quasi_cst = dominant_ratio[(dominant_ratio >= 0.99) & (~dominant_ratio.index.isin(PROTECTED_COLS))].index
    df = df.drop(columns=quasi_cst)

    # Feature engineering
    df["ANNUITY_CREDIT_RATIO"] = df["GROUP_ANNUITY_MEAN"] / df["GROUP_CREDIT_MEAN"]
    df["DEF_ESCALATION"] = df["DEF_60_CNT_SOCIAL_CIRCLE"] - df["DEF_30_CNT_SOCIAL_CIRCLE"]

    # Suppressions validées (mode full)
    cols_to_drop = [
        "AMT_CREDIT", "EXT_SOURCES_MEAN", "DAYS_EMPLOYED", "EXT_SOURCES_MIN",
        "INCOME_TO_EMPLOYED_RATIO", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "EXT_SOURCES_NANMEDIAN", "EMPLOYED_TO_BIRTH_RATIO", "AGE_RANGE",
        "REGION_RATING_CLIENT_W_CITY", "DAYS_LAST_PHONE_CHANGE", "INCOME_TO_BIRTH_RATIO",
        "DEF_60_CNT_SOCIAL_CIRCLE", "ID_TO_BIRTH_RATIO", "NEW_DOC_KURT", "GROUP_ANNUITY_MEAN",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df

# =============================================================================
# FEATURES CROISÉES : add_ratios_features
# =============================================================================

def add_ratios_features(df):
    """
    Crée des features de ratio croisant les données de la demande principale
    avec les informations agrégées du bureau et des demandes précédentes.

    Ces ratios permettent au modèle de comparer le comportement actuel du client
    à son historique de crédit.
    """
    #  Ratio crédit du bureau / revenu total 
    # Mesure la charge de crédit externe par rapport aux revenus
    df['BUREAU_INCOME_CREDIT_RATIO'] = df['BUREAU_AMT_CREDIT_SUM_MEAN'] / df['AMT_INCOME_TOTAL']
    df['BUREAU_ACTIVE_CREDIT_TO_INCOME_RATIO'] = df['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM'] / df['AMT_INCOME_TOTAL']

    #  Ratio crédit actuel / crédit des demandes précédentes approuvées 
    # Permet de voir si le client emprunte plus ou moins qu'habituellement
    df['CURRENT_TO_APPROVED_CREDIT_MIN_RATIO'] = df['APPROVED_AMT_CREDIT_MIN'] / df['AMT_CREDIT']
    df['CURRENT_TO_APPROVED_CREDIT_MAX_RATIO'] = df['APPROVED_AMT_CREDIT_MAX'] / df['AMT_CREDIT']
    df['CURRENT_TO_APPROVED_CREDIT_MEAN_RATIO'] = df['APPROVED_AMT_CREDIT_MEAN'] / df['AMT_CREDIT']

    #  Ratio annuité actuelle / annuités des demandes précédentes 
    # Mesure si la mensualité demandée est cohérente avec le passé
    df['CURRENT_TO_APPROVED_ANNUITY_MAX_RATIO'] = df['APPROVED_AMT_ANNUITY_MAX'] / df['AMT_ANNUITY']
    df['CURRENT_TO_APPROVED_ANNUITY_MEAN_RATIO'] = df['APPROVED_AMT_ANNUITY_MEAN'] / df['AMT_ANNUITY']
    # Ratios entre les paiements effectifs (installments) et l'annuité actuelle
    df['PAYMENT_MIN_TO_ANNUITY_RATIO'] = df['INS_AMT_PAYMENT_MIN'] / df['AMT_ANNUITY']
    df['PAYMENT_MAX_TO_ANNUITY_RATIO'] = df['INS_AMT_PAYMENT_MAX'] / df['AMT_ANNUITY']
    df['PAYMENT_MEAN_TO_ANNUITY_RATIO'] = df['INS_AMT_PAYMENT_MEAN'] / df['AMT_ANNUITY']

    #  Ratio crédit/annuité des demandes précédentes vs actuelle 
    # Compare la durée implicite du prêt (crédit ÷ annuité) avec le passé
    df['CTA_CREDIT_TO_ANNUITY_MAX_RATIO'] = df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MAX'] / df[
        'CREDIT_TO_ANNUITY_RATIO']
    df['CTA_CREDIT_TO_ANNUITY_MEAN_RATIO'] = df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MEAN'] / df[
        'CREDIT_TO_ANNUITY_RATIO']

    #  Ratios temporels 
    # Normalise les durées (en jours) par rapport à l'âge et à l'ancienneté du client
    df['DAYS_DECISION_MEAN_TO_BIRTH'] = df['APPROVED_DAYS_DECISION_MEAN'] / df['DAYS_BIRTH']
    df['DAYS_CREDIT_MEAN_TO_BIRTH'] = df['BUREAU_DAYS_CREDIT_MEAN'] / df['DAYS_BIRTH']
    df['DAYS_DECISION_MEAN_TO_EMPLOYED'] = df['APPROVED_DAYS_DECISION_MEAN'] / df['DAYS_EMPLOYED']
    df['DAYS_CREDIT_MEAN_TO_EMPLOYED'] = df['BUREAU_DAYS_CREDIT_MEAN'] / df['DAYS_EMPLOYED']

    # Les ratios peuvent produire +/-inf si le dénominateur vaut 0 → on les remplace par NaN
    df = replace_infinite_with_nan(df)
    return df


# =============================================================================
# PIPELINE DEMANDE PRINCIPALE : get_train
# =============================================================================

def get_train(path, num_rows=None):
    """
    Lit et prétraite le fichier application_train.csv.

    Ce fichier contient une ligne par demande de prêt en cours, avec les
    informations socio-démographiques du demandeur, les montants demandés,
    et la variable cible TARGET (1 = défaut de paiement, 0 = remboursé).

    Paramètre :
        path (str)     : Répertoire contenant les fichiers CSV.
        num_rows (int) : Nombre de lignes à lire (None = toutes).

    Retourne :
        df (pd.DataFrame) : DataFrame prétraité avec features engineered.
    """
    df = pd.read_csv(os.path.join(path, 'application_train.csv'), nrows=num_rows)

    #  Nettoyage des données aberrantes 
    df = df[df['CODE_GENDER'] != 'XNA']           # Supprime 4 individus avec genre inconnu ('XNA')
    df = df[df['AMT_INCOME_TOTAL'] < 20000000]    # Supprime un outlier extrême (117M€ dans le train, max 4M dans le test)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)          # 365243 = valeur sentinelle "non employé"
    df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan)  # 0 = pas de changement connu

    #  Features sur les documents fournis 
    # FLAG_DOC_X indique si le client a fourni le document X (0/1)
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)    # Nombre total de documents fournis
    df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1) # Kurtosis = mesure de la distribution des documents fournis

    #  Tranche d'âge 
    # Basée sur l'analyse de la corrélation TARGET=1 / âge
    df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))

    #  Features dérivées des scores externes (EXT_SOURCE_1/2/3) 
    # Ces scores sont fournis par des bureaux de crédit externes et sont très prédictifs
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    # Somme pondérée (SOURCE_3 compte double car plus prédictif)
    df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
    # Agrégations statistiques des 3 scores externes sur chaque ligne
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    #  Ratios financiers internes à la demande 
    # Rapport entre le montant emprunté et la mensualité → durée implicite du prêt
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    # Rapport entre le crédit accordé et la valeur du bien financé
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    # Ratios de charge financière par rapport aux revenus
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']

    # Ratios temporels : ancienneté professionnelle, véhicule, etc. par rapport à l'âge
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

    #  Statistiques de groupe 
    # Pour chaque combinaison (type d'organisation, niveau d'éducation, profession,
    # tranche d'âge, genre), calcule des statistiques de référence
    # → permet de comparer un individu à ses "pairs"
    group_cols = ['ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE', 'CODE_GENDER']
    df = do_median(df, group_cols, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_MEDIAN')
    df = do_std(df, group_cols, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_STD')
    df = do_mean(df, group_cols, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_MEAN')
    df = do_std(df, group_cols, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_STD')
    df = do_mean(df, group_cols, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_MEAN')
    df = do_std(df, group_cols, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_STD')
    df = do_mean(df, group_cols, 'AMT_CREDIT', 'GROUP_CREDIT_MEAN')
    df = do_mean(df, group_cols, 'AMT_ANNUITY', 'GROUP_ANNUITY_MEAN')
    df = do_std(df, group_cols, 'AMT_ANNUITY', 'GROUP_ANNUITY_STD')

    #  Encodage des variables catégorielles 
    # Les colonnes de type texte sont converties en entiers avec pd.factorize
    df, le_encoded_cols = label_encoder(df, None)

    #  Suppression des colonnes peu informatives 
    df = drop_application_columns(df)
    return df


def drop_application_columns(df):
    """
    Supprime les colonnes identifiées comme peu importantes par analyse
    de permutation feature importance.

    Ces colonnes ajoutent du bruit sans améliorer les performances du modèle.
    """
    drop_list = [
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
        'FLAG_OWN_REALTY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR',
        'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
        'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
        'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
        'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
        'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
        'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG', 'HOUSETYPE_MODE',
        'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE'
    ]
    # Supprime également la plupart des indicateurs de documents (peu discriminants)
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df


def get_age_label(days_birth):
    """
    Convertit un nombre de jours depuis la naissance (négatif dans le dataset)
    en un label de tranche d'âge (entier de 1 à 5).

    Tranches :
        1 : moins de 27 ans
        2 : 27 à 39 ans
        3 : 40 à 49 ans
        4 : 50 à 64 ans
        5 : 65 à 98 ans
        0 : valeur invalide (≥ 99 ans)
    """
    age_years = -days_birth / 365  # DAYS_BIRTH est négatif dans le dataset (jours avant aujourd'hui)
    if age_years < 27: return 1
    elif age_years < 40: return 2
    elif age_years < 50: return 3
    elif age_years < 65: return 4
    elif age_years < 99: return 5
    else: return 0


# =============================================================================
# PIPELINE BUREAU : get_bureau / get_bureau_balance
# =============================================================================

def get_bureau(path, num_rows=None):
    """
    Lit et agrège les données du Credit Bureau externe.

    Le Credit Bureau contient l'historique de tous les crédits du client
    auprès d'autres institutions financières (pas seulement Home Credit).

    Sources :
        bureau.csv          : Un ligne par crédit externe par client
        bureau_balance.csv  : Historique mensuel de chaque crédit externe

    Retourne :
        agg_bureau (pd.DataFrame) : Une ligne par client (SK_ID_CURR) avec
                                    des features agrégées.
    """
    bureau = pd.read_csv(os.path.join(path, 'bureau.csv'), nrows=num_rows)

    #  Features engineered sur chaque crédit 
    # Durée totale du crédit (du début à la date de fin prévue)
    bureau['CREDIT_DURATION'] = -bureau['DAYS_CREDIT'] + bureau['DAYS_CREDIT_ENDDATE']
    # Différence entre la date de fin prévue et la date de clôture réelle
    bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    # Taux d'endettement : dette restante / montant total du crédit
    bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
    # Différence absolue entre le crédit accordé et la dette restante
    bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    # Ratio crédit / annuité (durée implicite du remboursement)
    bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']

    #  Encodage one-hot des variables catégorielles du bureau 
    # Ex: CREDIT_ACTIVE → CREDIT_ACTIVE_Active, CREDIT_ACTIVE_Closed, etc.
    bureau, categorical_cols = one_hot_encoder(bureau, nan_as_category=False)

    #  Fusion avec les données mensuelles du bureau (bureau_balance) 
    bureau = bureau.merge(get_bureau_balance(path, num_rows), how='left', on='SK_ID_BUREAU')

    # Certaines colonnes STATUS peuvent être absentes après encodage one-hot si
    # aucun crédit n'avait ce statut dans le sous-ensemble chargé → on les initialise à 0
    for col in ['STATUS_0', 'STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5', 'STATUS_C', 'STATUS_X']:
        if col not in bureau.columns:
            bureau[col] = 0

    #  Indicateur de retard de paiement 
    # STATUS_1 à STATUS_5 indiquent des retards de 1 à 5 mois → on les cumule
    bureau['STATUS_12345'] = 0
    for i in range(1,6):
        bureau['STATUS_12345'] += bureau['STATUS_{}'.format(i)]

    #  Agrégation par durée de crédit (MONTHS_BALANCE_SIZE) 
    # Calcule les moyennes des features clés pour chaque "durée de crédit"
    # puis fusionne ces statistiques sur chaque ligne individuelle
    features = ['AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM',
        'AMT_CREDIT_SUM_DEBT', 'DEBT_PERCENTAGE', 'DEBT_CREDIT_DIFF', 'STATUS_0', 'STATUS_12345']
    agg_length = bureau.groupby('MONTHS_BALANCE_SIZE')[features].mean().reset_index()
    agg_length.rename({feat: 'LL_' + feat for feat in features}, axis=1, inplace=True)
    bureau = bureau.merge(agg_length, how='left', on='MONTHS_BALANCE_SIZE')
    del agg_length; gc.collect()

    #  Agrégation générale de tous les crédits par client 
    agg_bureau = group(bureau, 'BUREAU_', BUREAU_AGG)

    #  Agrégation séparée pour les crédits actifs et fermés 
    # Les crédits actifs et fermés ont des comportements très différents
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    agg_bureau = group_and_merge(active, agg_bureau, 'BUREAU_ACTIVE_', BUREAU_ACTIVE_AGG)
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    agg_bureau = group_and_merge(closed, agg_bureau, 'BUREAU_CLOSED_', BUREAU_CLOSED_AGG)
    del active, closed; gc.collect()

    #  Agrégation par type de crédit 
    # Crédit consommation, carte de crédit, hypothèque, prêt auto, microcrédit
    for credit_type in ['Consumer credit', 'Credit card', 'Mortgage', 'Car loan', 'Microloan']:
        type_df = bureau[bureau['CREDIT_TYPE_' + credit_type] == 1]
        prefix = 'BUREAU_' + credit_type.split(' ')[0].upper() + '_'
        agg_bureau = group_and_merge(type_df, agg_bureau, prefix, BUREAU_LOAN_TYPE_AGG)
        del type_df; gc.collect()

    #  Agrégation par fenêtre temporelle (6 et 12 derniers mois) 
    for time_frame in [6, 12]:
        prefix = "BUREAU_LAST{}M_".format(time_frame)
        time_frame_df = bureau[bureau['DAYS_CREDIT'] >= -30*time_frame]
        agg_bureau = group_and_merge(time_frame_df, agg_bureau, prefix, BUREAU_TIME_AGG)
        del time_frame_df; gc.collect()

    #  Impayé maximal du dernier crédit contracté 
    sort_bureau = bureau.sort_values(by=['DAYS_CREDIT'])
    gr = sort_bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].last().reset_index()
    gr.rename({'AMT_CREDIT_MAX_OVERDUE': 'BUREAU_LAST_LOAN_MAX_OVERDUE'}, inplace=True)
    agg_bureau = agg_bureau.merge(gr, on='SK_ID_CURR', how='left')

    #  Ratios dette / crédit total 
    # BUREAU_DEBT_OVER_CREDIT : taux global d'endettement sur tous les crédits
    agg_bureau['BUREAU_DEBT_OVER_CREDIT'] = \
        agg_bureau['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_AMT_CREDIT_SUM_SUM']
    # Version pour les crédits actifs uniquement
    agg_bureau['BUREAU_ACTIVE_DEBT_OVER_CREDIT'] = \
        agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM']
    return agg_bureau


def get_bureau_balance(path, num_rows=None):
    """
    Lit et agrège bureau_balance.csv.

    Ce fichier contient l'historique mensuel de chaque crédit externe
    (un enregistrement par mois et par crédit), incluant le statut de paiement.

    Retourne :
        bb_processed (pd.DataFrame) : Agrégé par SK_ID_BUREAU (identifiant du crédit externe).
    """
    bb = pd.read_csv(os.path.join(path, 'bureau_balance.csv'), nrows=num_rows)
    # One-hot encode le statut mensuel (STATUS : 0, 1, 2, ..., C, X)
    bb, categorical_cols = one_hot_encoder(bb, nan_as_category=False)
    # Calcule le taux moyen de chaque statut par crédit
    bb_processed = bb.groupby('SK_ID_BUREAU')[categorical_cols].mean().reset_index()
    # Ajoute les statistiques sur la durée de l'historique (en mois)
    agg = {'MONTHS_BALANCE': ['min', 'max', 'mean', 'size']}
    bb_processed = group_and_merge(bb, bb_processed, '', agg, 'SK_ID_BUREAU')
    del bb; gc.collect()
    return bb_processed


# =============================================================================
# PIPELINE DEMANDES PRÉCÉDENTES : get_previous_applications
# =============================================================================

def get_previous_applications(path, num_rows=None):
    """
    Lit et agrège previous_application.csv.

    Ce fichier contient toutes les demandes de crédit passées du client
    chez Home Credit (une ligne par demande), qu'elles aient été approuvées
    ou refusées.

    Retourne :
        agg_prev (pd.DataFrame) : Une ligne par client avec features agrégées.
    """
    prev = pd.read_csv(os.path.join(path, 'previous_application.csv'), nrows=num_rows)
    # Chargement du fichier des paiements pour calculer l'état des prêts actifs
    pay = pd.read_csv(os.path.join(path, 'installments_payments.csv'), nrows=num_rows)

    #  Encodage one-hot des colonnes catégorielles les plus importantes 
    ohe_columns = [
        'NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE', 'CHANNEL_TYPE',
        'NAME_TYPE_SUITE', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
        'NAME_PRODUCT_TYPE', 'NAME_CLIENT_TYPE']
    prev, categorical_cols = one_hot_encoder(prev, ohe_columns, nan_as_category=False)

    #  Features engineered 
    # Différence entre le montant demandé et le montant accordé
    prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    # Ratio demandé / accordé (>1 = le client a demandé plus qu'accordé)
    prev['APPLICATION_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Durée implicite du remboursement (en mois)
    prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    # Part de l'apport personnel par rapport au crédit total
    prev['DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']
    # Taux d'intérêt simplifié : (total payé / montant emprunté - 1) / nb mensualités
    total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['SIMPLE_INTERESTS'] = (total_payment / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']

    #  Analyse des prêts actifs (approuvés mais pas encore terminés) 
    # DAYS_LAST_DUE = 365243 est la valeur sentinelle pour "prêt toujours en cours"
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    active_df = approved[approved['DAYS_LAST_DUE'] == 365243]

    # Calcule combien a déjà été remboursé sur ces prêts actifs
    active_pay = pay[pay['SK_ID_PREV'].isin(active_df['SK_ID_PREV'])]
    active_pay_agg = active_pay.groupby('SK_ID_PREV')[['AMT_INSTALMENT', 'AMT_PAYMENT']].sum()
    active_pay_agg.reset_index(inplace=True)

    # Différence entre ce qui était dû et ce qui a été payé (retard cumulé)
    active_pay_agg['INSTALMENT_PAYMENT_DIFF'] = active_pay_agg['AMT_INSTALMENT'] - active_pay_agg['AMT_PAYMENT']

    # Fusionne avec les prêts actifs pour obtenir la dette restante et le taux de remboursement
    active_df = active_df.merge(active_pay_agg, on='SK_ID_PREV', how='left')
    active_df['REMAINING_DEBT'] = active_df['AMT_CREDIT'] - active_df['AMT_PAYMENT']
    active_df['REPAYMENT_RATIO'] = active_df['AMT_PAYMENT'] / active_df['AMT_CREDIT']

    # Agrégation des prêts actifs par client
    active_agg_df = group(active_df, 'PREV_ACTIVE_', PREVIOUS_ACTIVE_AGG)
    # Ratio global de remboursement des prêts actifs
    active_agg_df['TOTAL_REPAYMENT_RATIO'] = active_agg_df['PREV_ACTIVE_AMT_PAYMENT_SUM'] / \
                                             active_agg_df['PREV_ACTIVE_AMT_CREDIT_SUM']
    del active_pay, active_pay_agg, active_df; gc.collect()

    #  Nettoyage : remplacement des valeurs sentinelles par NaN 
    # 365243 indique une date non renseignée dans ce contexte
    prev['DAYS_FIRST_DRAWING'] = prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan)
    prev['DAYS_FIRST_DUE'] = prev['DAYS_FIRST_DUE'].replace(365243, np.nan)
    prev['DAYS_LAST_DUE_1ST_VERSION'] = prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan)
    prev['DAYS_LAST_DUE'] = prev['DAYS_LAST_DUE'].replace(365243, np.nan)
    prev['DAYS_TERMINATION'] = prev['DAYS_TERMINATION'].replace(365243, np.nan)

    # Différence entre la date de fin prévue initialement et celle révisée
    prev['DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']
    approved['DAYS_LAST_DUE_DIFF'] = approved['DAYS_LAST_DUE_1ST_VERSION'] - approved['DAYS_LAST_DUE']

    #  Agrégation générale de toutes les demandes précédentes 
    categorical_agg = {key: ['mean'] for key in categorical_cols}  # Moyenne des colonnes one-hot
    agg_prev = group(prev, 'PREV_', {**PREVIOUS_AGG, **categorical_agg})

    # Fusionne avec les agrégations des prêts actifs
    agg_prev = agg_prev.merge(active_agg_df, how='left', on='SK_ID_CURR')
    del active_agg_df; gc.collect()

    #  Agrégations séparées par statut (approuvé / refusé) 
    agg_prev = group_and_merge(approved, agg_prev, 'APPROVED_', PREVIOUS_APPROVED_AGG)
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    agg_prev = group_and_merge(refused, agg_prev, 'REFUSED_', PREVIOUS_REFUSED_AGG)
    del approved, refused; gc.collect()

    #  Agrégations par type de prêt (consommation / cash) 
    for loan_type in ['Consumer loans', 'Cash loans']:
        type_df = prev[prev['NAME_CONTRACT_TYPE_{}'.format(loan_type)] == 1]
        prefix = 'PREV_' + loan_type.split(" ")[0] + '_'
        agg_prev = group_and_merge(type_df, agg_prev, prefix, PREVIOUS_LOAN_TYPE_AGG)
        del type_df; gc.collect()

    #  Analyse des prêts avec retards de paiement 
    # LATE_PAYMENT = 1 si le paiement a été effectué après la date d'échéance
    pay['LATE_PAYMENT'] = pay['DAYS_ENTRY_PAYMENT'] - pay['DAYS_INSTALMENT']
    pay['LATE_PAYMENT'] = pay['LATE_PAYMENT'].apply(lambda x: 1 if x > 0 else 0)
    dpd_id = pay[pay['LATE_PAYMENT'] > 0]['SK_ID_PREV'].unique()
    # Agrégation des caractéristiques des prêts ayant connu des retards
    agg_dpd = group_and_merge(prev[prev['SK_ID_PREV'].isin(dpd_id)], agg_prev,
                                    'PREV_LATE_', PREVIOUS_LATE_PAYMENTS_AGG)
    del agg_dpd, dpd_id; gc.collect()

    #  Agrégation par fenêtre temporelle récente (12 et 24 derniers mois) 
    for time_frame in [12, 24]:
        time_frame_df = prev[prev['DAYS_DECISION'] >= -30*time_frame]
        prefix = 'PREV_LAST{}M_'.format(time_frame)
        agg_prev = group_and_merge(time_frame_df, agg_prev, prefix, PREVIOUS_TIME_AGG)
        del time_frame_df; gc.collect()
    del prev; gc.collect()
    return agg_prev


# =============================================================================
# PIPELINE POS-CASH : get_pos_cash
# =============================================================================

def get_pos_cash(path, num_rows=None):
    """
    Lit et agrège POS_CASH_balance.csv.

    Ce fichier contient le statut mensuel des prêts à la consommation et des
    contrats POS (Point Of Sale = achats à crédit en magasin) précédents.

    Retourne :
        pos_agg (pd.DataFrame) : Agrégé par client (SK_ID_CURR).
    """
    pos = pd.read_csv(os.path.join(path, 'POS_CASH_balance.csv'), nrows=num_rows)
    pos, categorical_cols = one_hot_encoder(pos, nan_as_category=False)

    # Indicateur binaire : retard de paiement ce mois-ci (SK_DPD > 0 = jours de retard)
    pos['LATE_PAYMENT'] = pos['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)

    #  Agrégation globale par client 
    categorical_agg = {key: ['mean'] for key in categorical_cols}
    pos_agg = group(pos, 'POS_', {**POS_CASH_AGG, **categorical_agg})

    #  Features sur l'état d'avancement des prêts 
    # Trie par prêt et par mois pour accéder au premier/dernier enregistrement
    sort_pos = pos.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
    gp = sort_pos.groupby('SK_ID_PREV')
    df = pd.DataFrame()
    df['SK_ID_CURR'] = gp['SK_ID_CURR'].first()
    df['MONTHS_BALANCE_MAX'] = gp['MONTHS_BALANCE'].max()
    # Taux moyen de prêts arrivés à complétion
    df['POS_LOAN_COMPLETED_MEAN'] = gp['NAME_CONTRACT_STATUS_Completed'].mean()
    # Indicateur : prêt remboursé avant l'échéance initiale
    # (CNT_INSTALMENT initial > CNT_INSTALMENT final = le nb d'échéances a diminué)
    df['POS_COMPLETED_BEFORE_MEAN'] = gp['CNT_INSTALMENT'].first() - gp['CNT_INSTALMENT'].last()
    df['POS_COMPLETED_BEFORE_MEAN'] = df.apply(lambda x: 1 if x['POS_COMPLETED_BEFORE_MEAN'] > 0
                                                and x['POS_LOAN_COMPLETED_MEAN'] > 0 else 0, axis=1)
    # Nombre et ratio d'échéances restantes (futures)
    df['POS_REMAINING_INSTALMENTS'] = gp['CNT_INSTALMENT_FUTURE'].last()
    df['POS_REMAINING_INSTALMENTS_RATIO'] = gp['CNT_INSTALMENT_FUTURE'].last() / gp['CNT_INSTALMENT'].last()

    # Agrège par client et fusionne
    df_gp = df.groupby('SK_ID_CURR').sum().reset_index()
    df_gp.drop(['MONTHS_BALANCE_MAX'], axis=1, inplace=True)
    pos_agg = pd.merge(pos_agg, df_gp, on='SK_ID_CURR', how='left')
    del df, gp, df_gp, sort_pos; gc.collect()

    #  Taux de retard sur les 3 demandes les plus récentes 
    pos = do_sum(pos, ['SK_ID_PREV'], 'LATE_PAYMENT', 'LATE_PAYMENT_SUM')
    # Identifie le dernier mois de chaque prêt
    last_month_df = pos.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()
    # Prend les 3 prêts les plus récents par client
    sort_pos = pos.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
    gp = sort_pos.iloc[last_month_df].groupby('SK_ID_CURR').tail(3)
    gp_mean = gp.groupby('SK_ID_CURR').mean().reset_index()
    pos_agg = pd.merge(pos_agg, gp_mean[['SK_ID_CURR','LATE_PAYMENT_SUM']], on='SK_ID_CURR', how='left')

    #  Suppression des colonnes catégorielles peu utiles 
    drop_features = [
        'POS_NAME_CONTRACT_STATUS_Canceled_MEAN', 'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN',
        'POS_NAME_CONTRACT_STATUS_XNA_MEAN']
    pos_agg.drop([f for f in drop_features if f in pos_agg.columns], axis=1, inplace=True)
    return pos_agg


# =============================================================================
# PIPELINE PAIEMENTS D'ÉCHÉANCES : get_installment_payments
# =============================================================================

def get_installment_payments(path, num_rows=None):
    """
    Lit et agrège installments_payments.csv.

    Ce fichier contient l'historique détaillé de chaque paiement d'échéance,
    pour chaque prêt précédent ou en cours du client.

    Retourne :
        pay_agg (pd.DataFrame) : Agrégé par client (SK_ID_CURR).
    """
    pay = pd.read_csv(os.path.join(path, 'installments_payments.csv'), nrows=num_rows)

    #  Regroupement des paiements par numéro d'échéance 
    # Un client peut payer en plusieurs fois pour une même échéance → on les somme
    pay = do_sum(pay, ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], 'AMT_PAYMENT', 'AMT_PAYMENT_GROUPED')
    pay['PAYMENT_DIFFERENCE'] = pay['AMT_INSTALMENT'] - pay['AMT_PAYMENT_GROUPED']  # Solde non payé
    pay['PAYMENT_RATIO'] = pay['AMT_INSTALMENT'] / pay['AMT_PAYMENT_GROUPED']       # Taux de couverture
    pay['PAID_OVER_AMOUNT'] = pay['AMT_PAYMENT'] - pay['AMT_INSTALMENT']             # Surpayement éventuel
    pay['PAID_OVER'] = (pay['PAID_OVER_AMOUNT'] > 0).astype(int)                     # Indicateur surpayement

    #  Indicateurs de ponctualité de paiement 
    # DPD : Days Past Due = nombre de jours de retard (0 si payé à temps ou en avance)
    pay['DPD'] = pay['DAYS_ENTRY_PAYMENT'] - pay['DAYS_INSTALMENT']
    pay['DPD'] = pay['DPD'].apply(lambda x: 0 if x <= 0 else x)
    # DBD : Days Before Due = nombre de jours d'avance (0 si en retard ou à temps)
    pay['DBD'] = pay['DAYS_INSTALMENT'] - pay['DAYS_ENTRY_PAYMENT']
    pay['DBD'] = pay['DBD'].apply(lambda x: 0 if x <= 0 else x)
    # Indicateur binaire : paiement effectué avant l'échéance
    pay['LATE_PAYMENT'] = pay['DBD'].apply(lambda x: 1 if x > 0 else 0)

    #  Ratios de paiement 
    pay['INSTALMENT_PAYMENT_RATIO'] = pay['AMT_PAYMENT'] / pay['AMT_INSTALMENT']
    # Ratio de paiement uniquement pour les paiements tardifs (0 sinon)
    pay['LATE_PAYMENT_RATIO'] = pay.apply(lambda x: x['INSTALMENT_PAYMENT_RATIO'] if x['LATE_PAYMENT'] == 1 else 0, axis=1)
    # Indicateur : retard significatif (ratio > 5% de l'échéance)
    pay['SIGNIFICANT_LATE_PAYMENT'] = pay['LATE_PAYMENT_RATIO'].apply(lambda x: 1 if x > 0.05 else 0)
    # Indicateurs de retard par seuil de jours (7 jours et 15 jours)
    pay['DPD_7'] = pay['DPD'].apply(lambda x: 1 if x >= 7 else 0)
    pay['DPD_15'] = pay['DPD'].apply(lambda x: 1 if x >= 15 else 0)

    #  Agrégation globale par client 
    pay_agg = group(pay, 'INS_', INSTALLMENTS_AGG)

    #  Agrégation sur les paiements récents (36 et 60 derniers mois) 
    for months in [36, 60]:
        recent_prev_id = pay[pay['DAYS_INSTALMENT'] >= -30*months]['SK_ID_PREV'].unique()
        pay_recent = pay[pay['SK_ID_PREV'].isin(recent_prev_id)]
        prefix = 'INS_{}M_'.format(months)
        pay_agg = group_and_merge(pay_recent, pay_agg, prefix, INSTALLMENTS_TIME_AGG)

    #  Features de tendance sur les K dernières échéances 
    # Utilise une régression linéaire pour détecter si les retards augmentent ou diminuent
    group_features = ['SK_ID_CURR', 'SK_ID_PREV', 'DPD', 'LATE_PAYMENT',
                      'PAID_OVER_AMOUNT', 'PAID_OVER', 'DAYS_INSTALMENT']
    gp = pay[group_features].groupby('SK_ID_CURR')
    func = partial(trend_in_last_k_instalment_features, periods=INSTALLMENTS_LAST_K_TREND_PERIODS)
    g = parallel_apply(gp, func, index_name='SK_ID_CURR', chunk_size=10000).reset_index()
    pay_agg = pay_agg.merge(g, on='SK_ID_CURR', how='left')

    #  Features sur le dernier prêt uniquement 
    g = parallel_apply(gp, installments_last_loan_features, index_name='SK_ID_CURR', chunk_size=10000).reset_index()
    pay_agg = pay_agg.merge(g, on='SK_ID_CURR', how='left')
    return pay_agg


def trend_in_last_k_instalment_features(gr, periods):
    """
    Calcule la pente (tendance) des retards et surpaiements sur les K
    dernières échéances de chaque client.

    Une pente positive de DPD indique que les retards s'aggravent → signal de risque.
    Une pente négative indique une amélioration du comportement de paiement.

    Paramètres :
        gr      : Groupe de paiements d'un client (DataFrame)
        periods : Liste de périodes K (ex: [12, 24, 60, 120])

    Retourne :
        features (dict) : Pentes calculées pour chaque période et chaque feature.
    """
    gr_ = gr.copy()
    # Trie par date d'échéance décroissante → les plus récentes en premier
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]  # Garde seulement les K dernières échéances
        features = add_trend_feature(features, gr_period, 'DPD',
                                           '{}_TREND_'.format(period))
        features = add_trend_feature(features, gr_period, 'PAID_OVER_AMOUNT',
                                           '{}_TREND_'.format(period))
    return features


def installments_last_loan_features(gr):
    """
    Calcule des statistiques sur les paiements du dernier prêt actif du client.

    Se concentre sur le prêt le plus récent (SK_ID_PREV le plus récent) pour
    capturer le comportement actuel, pas seulement l'historique global.

    Retourne :
        features (dict) : Statistiques (sum, mean, max, std, etc.) sur DPD,
                          LATE_PAYMENT et PAID_OVER_AMOUNT pour le dernier prêt.
    """
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    # Identifie l'ID du prêt le plus récent (première ligne après tri décroissant)
    last_installment_id = gr_['SK_ID_PREV'].iloc[0]
    gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]  # Filtre sur ce prêt uniquement

    features = {}
    features = add_features_in_group(features, gr_, 'DPD',
                                     ['sum', 'mean', 'max', 'std'],
                                     'LAST_LOAN_')
    features = add_features_in_group(features, gr_, 'LATE_PAYMENT',
                                     ['count', 'mean'],
                                     'LAST_LOAN_')
    features = add_features_in_group(features, gr_, 'PAID_OVER_AMOUNT',
                                     ['sum', 'mean', 'max', 'min', 'std'],
                                     'LAST_LOAN_')
    features = add_features_in_group(features, gr_, 'PAID_OVER',
                                     ['count', 'mean'],
                                     'LAST_LOAN_')
    return features


# =============================================================================
# PIPELINE CARTES DE CRÉDIT : get_credit_card
# =============================================================================

def get_credit_card(path, num_rows=None):
    """
    Lit et agrège credit_card_balance.csv.

    Ce fichier contient les soldes mensuels de chaque carte de crédit
    précédente ou en cours du client.

    Retourne :
        cc_agg (pd.DataFrame) : Agrégé par client (SK_ID_CURR).
    """
    cc = pd.read_csv(os.path.join(path, 'credit_card_balance.csv'), nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=False)
    # Correction d'une faute de frappe dans le nom de colonne du CSV source
    cc.rename(columns={'AMT_RECIVABLE': 'AMT_RECEIVABLE'}, inplace=True)

    #  Features engineered 
    # Taux d'utilisation du plafond de crédit (balance / limite)
    cc['LIMIT_USE'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    # Ratio entre le paiement effectué et le minimum dû
    cc['PAYMENT_DIV_MIN'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']
    # Indicateur de retard (SK_DPD = jours de retard)
    cc['LATE_PAYMENT'] = cc['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    # Ratio retraits ATM / limite de crédit (mesure le type d'utilisation)
    cc['DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']

    #  Agrégation globale par client 
    cc_card_agg = {k: v for k, v in CREDIT_CARD_AGG.items() if k in cc.columns}
    cc_agg = cc.groupby('SK_ID_CURR').agg(cc_card_agg)
    # Renomme les colonnes : CC_NOMCOLONNE_AGREGATION (ex: CC_AMT_BALANCE_MAX)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg.reset_index(inplace=True)

    #  Features sur le dernier mois de chaque carte 
    # Prend la ligne correspondant au mois le plus récent de chaque carte
    last_ids = cc.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()
    last_months_df = cc[cc.index.isin(last_ids)]
    cc_agg = group_and_merge(last_months_df, cc_agg, 'CC_LAST_', {'AMT_BALANCE': ['mean', 'max']})

    #  Agrégation sur les périodes récentes (12, 24, 48 derniers mois) 
    for months in [12, 24, 48]:
        cc_prev_id = cc[cc['MONTHS_BALANCE'] >= -months]['SK_ID_PREV'].unique()
        cc_recent = cc[cc['SK_ID_PREV'].isin(cc_prev_id)]
        prefix = 'INS_{}M_'.format(months)
        cc_agg = group_and_merge(cc_recent, cc_agg, prefix, CREDIT_CARD_TIME_AGG)
    return cc_agg


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

@contextmanager
def timer(name):
    """
    Gestionnaire de contexte pour mesurer le temps d'exécution d'un bloc.

    Utilisation :
        with timer("nom de l'étape"):
            # code à mesurer

    Affiche : "nom de l'étape - done in Xs"
    """
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(name, time.time() - t0))


def group(df_to_agg, prefix, aggregations, aggregate_by='SK_ID_CURR'):
    """
    Agrège un DataFrame par une colonne clé et renomme les colonnes résultantes.

    Paramètres :
        df_to_agg    : DataFrame source à agréger
        prefix       : Préfixe à ajouter aux noms de colonnes (ex: 'BUREAU_')
        aggregations : Dictionnaire {colonne: [liste d'agrégations]}
                       ex: {'AMT_CREDIT': ['mean', 'max']}
        aggregate_by : Colonne de regroupement (défaut: 'SK_ID_CURR')

    Retourne :
        DataFrame agrégé avec colonnes nommées PREFIX_COLONNE_AGREGATION
        ex: BUREAU_AMT_CREDIT_MEAN, BUREAU_AMT_CREDIT_MAX
    """
    # Filtre les colonnes demandées qui n'existent pas dans le DataFrame
    aggregations = {k: v for k, v in aggregations.items() if k in df_to_agg.columns}
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()


def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by='SK_ID_CURR'):
    """
    Agrège df_to_agg puis fusionne (left join) le résultat dans df_to_merge.

    Raccourci combinant group() + merge() en une seule opération.
    """
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by=aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on=aggregate_by)


def do_mean(df, group_cols, counted, agg_name):
    """
    Calcule la moyenne de `counted` groupée par `group_cols` et l'ajoute au DataFrame.

    Utile pour créer des features de référence de groupe (ex : revenu moyen
    du groupe socio-démographique du client).
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_median(df, group_cols, counted, agg_name):
    """
    Calcule la médiane de `counted` groupée par `group_cols` et l'ajoute au DataFrame.

    La médiane est plus robuste aux outliers que la moyenne.
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_std(df, group_cols, counted, agg_name):
    """
    Calcule l'écart-type de `counted` groupé par `group_cols` et l'ajoute au DataFrame.

    Mesure la dispersion au sein du groupe → un client avec une valeur très
    éloignée de la moyenne de son groupe peut être un signal de risque.
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_sum(df, group_cols, counted, agg_name):
    """
    Calcule la somme de `counted` groupée par `group_cols` et l'ajoute au DataFrame.

    Utile pour agréger des indicateurs binaires (ex : nombre total de retards).
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """
    Applique un encodage one-hot aux colonnes catégorielles.

    Chaque valeur unique d'une colonne catégorielle devient une nouvelle colonne binaire.
    Ex: CREDIT_ACTIVE = ['Active', 'Closed'] → CREDIT_ACTIVE_Active + CREDIT_ACTIVE_Closed

    Paramètres :
        df                  : DataFrame source
        categorical_columns : Liste de colonnes à encoder (None = toutes les colonnes texte)
        nan_as_category     : Si True, crée une colonne séparée pour les valeurs NaN

    Retourne :
        df               : DataFrame avec les nouvelles colonnes one-hot
        categorical_cols : Liste des noms des nouvelles colonnes créées
    """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'str']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns


def label_encoder(df, categorical_columns=None):
    """
    Encode les variables catégorielles en entiers avec pd.factorize.

    Contrairement à l'encodage one-hot, cette méthode ne crée pas de nouvelles colonnes :
    elle remplace les valeurs texte par des entiers (0, 1, 2, ...).
    Utilisé ici pour les colonnes binaires ou ordinales.

    Retourne :
        df               : DataFrame avec colonnes encodées
        categorical_cols : Liste des colonnes qui ont été encodées
    """
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'str']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df, categorical_columns


def add_features(feature_name, aggs, features, feature_names, groupby):
    """
    Calcule et fusionne des agrégations statistiques avancées dans un DataFrame features.

    Supporte les agrégations standard (mean, max, etc.) ainsi que :
        'kurt' : kurtosis (mesure des queues de distribution)
        'iqr'  : intervalle interquartile (robustesse aux outliers)

    Paramètres :
        feature_name  : Nom de la feature à agréger
        aggs          : Liste des agrégations à appliquer
        features      : DataFrame dans lequel fusionner les résultats
        feature_names : Liste des noms de features (étendue en place)
        groupby       : Objet GroupBy pandas

    Retourne :
        (features, feature_names) : DataFrame mis à jour et noms de colonnes ajoutés
    """
    feature_names.extend(['{}_{}'.format(feature_name, agg) for agg in aggs])

    for agg in aggs:
        if agg == 'kurt':
            agg_func = kurtosis
        elif agg == 'iqr':
            agg_func = iqr
        else:
            agg_func = agg  # Nom de méthode pandas standard (ex: 'mean', 'max')

        g = groupby[feature_name].agg(agg_func).reset_index().rename(index=str,
                                                                     columns={feature_name: '{}_{}'.format(feature_name, agg)})
        features = features.merge(g, on='SK_ID_CURR', how='left')
    return features, feature_names


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    """
    Calcule des statistiques sur un groupe (sous-DataFrame) et les stocke dans un dict.

    Utilisé dans les fonctions de traitement par groupe (ex: installments_last_loan_features).

    Paramètres :
        features     : Dictionnaire de résultats à compléter
        gr_          : Sous-DataFrame du groupe courant
        feature_name : Nom de la colonne à agréger
        aggs         : Liste des agrégations : 'sum', 'mean', 'max', 'min', 'std',
                       'count', 'skew', 'kurt', 'iqr', 'median'
        prefix       : Préfixe des clés dans le dictionnaire (ex: 'LAST_LOAN_')

    Retourne :
        features (dict) : Dictionnaire enrichi des statistiques calculées
    """
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features


def add_trend_feature(features, gr, feature_name, prefix):
    """
    Calcule la tendance (pente) d'une feature sur une série temporelle.

    Utilise une régression linéaire simple : la pente (coef_[0]) indique
    si la feature augmente ou diminue au fil du temps.

    Ex : une pente positive de DPD indique que les retards s'aggravent.

    Paramètres :
        features     : Dictionnaire des features à enrichir
        gr           : Groupe (DataFrame trié par temps)
        feature_name : Colonne sur laquelle calculer la tendance
        prefix       : Préfixe du nom de la feature (ex: '12_TREND_')

    Retourne :
        features (dict) : Dictionnaire avec la pente ajoutée (NaN si calcul impossible)
    """
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)  # Variable temporelle (indice de position)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]  # Pente de la droite de régression
    except:
        trend = np.nan  # Retourne NaN si le calcul échoue (ex: groupe vide ou uniforme)
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def parallel_apply(groups, func, index_name='Index', num_workers=0, chunk_size=100000):
    """
    Applique une fonction à chaque groupe d'un GroupBy en parallèle.

    Divise les groupes en chunks et utilise un Pool de processus pour
    accélérer les calculs sur les gros datasets.

    Paramètres :
        groups     : Objet GroupBy pandas
        func       : Fonction à appliquer à chaque groupe (retourne un dict)
        index_name : Nom de l'index dans le DataFrame résultant
        num_workers: Nombre de processus parallèles (0 = utilise NUM_THREADS)
        chunk_size : Nombre de groupes traités par batch

    Retourne :
        features (pd.DataFrame) : DataFrame avec une ligne par groupe
    """
    if num_workers <= 0: num_workers = NUM_THREADS
    indeces, features = [], []
    for index_chunk, groups_chunk in chunk_groups(groups, chunk_size):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)  # Traitement parallèle
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def chunk_groups(groupby_object, chunk_size):
    """
    Générateur qui divise un objet GroupBy en chunks de taille fixe.

    Permet de traiter de très grands GroupBy en mémoire contrôlée.

    Paramètres :
        groupby_object : GroupBy pandas à découper
        chunk_size     : Nombre de groupes par chunk

    Yields :
        (index_chunk, group_chunk) : Tuple (liste d'index, liste de DataFrames)
    """
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)
        # Yield dès que le chunk est plein, ou à la fin des groupes
        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


def reduce_memory(df):
    """
    Réduit l'empreinte mémoire du DataFrame en utilisant les types numériques
    les plus petits possibles.

    Pour chaque colonne numérique, vérifie les valeurs min/max et choisit
    le type le plus compact :
        - int64 → int8/int16/int32 si les valeurs le permettent
        - float64 → float16/float32 si les valeurs le permettent

    Affiche la réduction de mémoire obtenue.

    Retourne :
        df (pd.DataFrame) : Même DataFrame avec des types optimisés
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Initial df memory usage is {:.2f} MB for {} columns'
          .format(start_mem, len(df.columns)))

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:  # Ne touche pas les colonnes texte
            cmin = df[col].min()
            cmax = df[col].max()
            if str(col_type)[:3] == 'int':
                # Choisit le plus petit type entier compatible
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Choisit le plus petit type flottant compatible
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    memory_reduction = 100 * (start_mem - end_mem) / start_mem
    print('Final memory usage is: {:.2f} MB - decreased by {:.1f}%'.format(end_mem, memory_reduction))
    return df


def replace_infinite_with_nan(df):
    """
    Remplace toutes les valeurs +inf et -inf par NaN dans les colonnes numériques.

    Ces valeurs infinies apparaissent lors des divisions par zéro dans les
    calculs de ratios. Les modèles ML ne peuvent pas traiter les infinis,
    donc on les convertit en NaN (valeurs manquantes gérées par imputation).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df

    inf_mask = np.isinf(df[numeric_cols].to_numpy())
    inf_count = int(inf_mask.sum())
    if inf_count > 0:
        print(f"Replacing {inf_count} infinite values with NaN")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def sanitize_feature_names(df):
    """
    Normalise les noms de colonnes pour qu'ils ne contiennent que des
    caractères alphanumériques et underscores (ASCII-safe).

    Certains encodages one-hot créent des noms avec des espaces ou caractères
    spéciaux (ex: 'CREDIT_TYPE_Consumer credit') qui sont incompatibles avec
    certains frameworks ML (notamment LightGBM).

    Traitement :
        1. Remplace tous les caractères non alphanumériques par '_'
        2. Supprime les underscores multiples consécutifs
        3. Résout les doublons de noms en ajoutant un suffixe numérique

    Affiche le nombre de colonnes renommées ou dédoublonnées.
    """
    original_cols = [str(col) for col in df.columns]
    cleaned_cols = []
    changed_count = 0

    for col in original_cols:
        # Remplace tout caractère non alphanumérique/underscore par '_'
        clean = re.sub(r'[^0-9A-Za-z_]+', '_', col)
        # Réduit les séquences d'underscores multiples à un seul
        clean = re.sub(r'__+', '_', clean).strip('_')
        if not clean:
            clean = 'feature'  # Nom de repli si la colonne devient vide après nettoyage
        if clean != col:
            changed_count += 1
        cleaned_cols.append(clean)

    # Résolution des doublons (deux colonnes différentes pourraient avoir le même nom nettoyé)
    unique_cols = []
    seen = {}
    duplicate_resolved = 0
    for col in cleaned_cols:
        count = seen.get(col, 0)
        if count == 0:
            unique_cols.append(col)
        else:
            unique_cols.append(f"{col}_{count}")  # Ajoute un suffixe numérique pour différencier
            duplicate_resolved += 1
        seen[col] = count + 1

    df.columns = unique_cols

    if changed_count > 0 or duplicate_resolved > 0:
        print(
            f"Sanitized feature names: {changed_count} renamed, "
            f"{duplicate_resolved} duplicates disambiguated"
        )
    return df


# =============================================================================
# CONFIGURATIONS GLOBALES
# =============================================================================

# Nombre de threads pour le traitement parallèle
NUM_THREADS = 4

# Répertoire contenant tous les fichiers CSV source
DATA_DIRECTORY = "data/raw/"

# Périodes (en nombre d'échéances) pour les features de tendance des paiements
# Ex: les 12, 24, 60 et 120 dernières échéances
INSTALLMENTS_LAST_K_TREND_PERIODS = [12, 24, 60, 120]


# =============================================================================
# DICTIONNAIRES D'AGRÉGATIONS
# Ces dictionnaires définissent quelles statistiques calculer sur quelles colonnes
# lors des agrégations par client. Format : {nom_colonne: [liste_d_agrégations]}
# =============================================================================

#  Agrégations générales sur tous les crédits du bureau 
BUREAU_AGG = {
    'SK_ID_BUREAU': ['nunique'],              # Nombre de crédits externes distincts
    'DAYS_CREDIT': ['min', 'max', 'mean'],    # Ancienneté des crédits (en jours)
    'DAYS_CREDIT_ENDDATE': ['min', 'max'],    # Dates de fin prévues
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'], # Impayé maximal historique
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'], # Montants des crédits
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],  # Dettes restantes
    'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'], # Montants en retard
    'AMT_ANNUITY': ['mean'],                  # Mensualité moyenne
    'DEBT_CREDIT_DIFF': ['mean', 'sum'],      # Différence crédit - dette
    'MONTHS_BALANCE_MEAN': ['mean', 'var'],   # Moyenne des balances mensuelles
    'MONTHS_BALANCE_SIZE': ['mean', 'sum'],   # Durée de l'historique mensuel
    'STATUS_0': ['mean'],        # % de mois sans retard
    'STATUS_1': ['mean'],        # % de mois avec 1 mois de retard
    'STATUS_12345': ['mean'],    # % de mois avec au moins 1 mois de retard
    'STATUS_C': ['mean'],        # % de mois où le crédit était clôturé
    'STATUS_X': ['mean'],        # % de mois avec statut inconnu
    'CREDIT_ACTIVE_Active': ['mean'],           # Proportion de crédits actifs
    'CREDIT_ACTIVE_Closed': ['mean'],           # Proportion de crédits fermés
    'CREDIT_ACTIVE_Sold': ['mean'],             # Proportion de crédits vendus
    'CREDIT_TYPE_Consumer credit': ['mean'],    # Proportion par type de crédit
    'CREDIT_TYPE_Credit card': ['mean'],
    'CREDIT_TYPE_Car loan': ['mean'],
    'CREDIT_TYPE_Mortgage': ['mean'],
    'CREDIT_TYPE_Microloan': ['mean'],
    'LL_AMT_CREDIT_SUM_OVERDUE': ['mean'],  # Features basées sur la durée du crédit (LL = loan length)
    'LL_DEBT_CREDIT_DIFF': ['mean'],
    'LL_STATUS_12345': ['mean'],
}

#  Agrégations sur les crédits actifs uniquement 
BUREAU_ACTIVE_AGG = {
    'DAYS_CREDIT': ['max', 'mean'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['max', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean'],
    'DAYS_CREDIT_UPDATE': ['min', 'mean'],  # Date de dernière mise à jour du crédit
    'DEBT_PERCENTAGE': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean'],
    'CREDIT_TO_ANNUITY_RATIO': ['mean'],
    'MONTHS_BALANCE_MEAN': ['mean', 'var'],
    'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
}

#  Agrégations sur les crédits fermés uniquement 
BUREAU_CLOSED_AGG = {
    'DAYS_CREDIT': ['max', 'var'],
    'DAYS_CREDIT_ENDDATE': ['max'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'sum'],
    'DAYS_CREDIT_UPDATE': ['max'],
    'ENDDATE_DIF': ['mean'],   # Différence entre date de fin prévue et réelle
    'STATUS_12345': ['mean'],
}

#  Agrégations par type de prêt (consommation, carte, hypothèque, auto, micro) 
BUREAU_LOAN_TYPE_AGG = {
    'DAYS_CREDIT': ['mean', 'max'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
    'AMT_CREDIT_SUM': ['mean', 'max'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'max'],
    'DEBT_PERCENTAGE': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean'],
    'DAYS_CREDIT_ENDDATE': ['max'],
}

#  Agrégations sur les crédits récents (6 ou 12 derniers mois) 
BUREAU_TIME_AGG = {
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    'DEBT_PERCENTAGE': ['mean'],
    'DEBT_CREDIT_DIFF': ['mean'],
    'STATUS_0': ['mean'],
    'STATUS_12345': ['mean'],
}

#  Agrégations sur toutes les demandes précédentes 
PREVIOUS_AGG = {
    'SK_ID_PREV': ['nunique'],            # Nombre de demandes précédentes
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_DOWN_PAYMENT': ['max', 'mean'],
    'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],  # Heure de la demande
    'RATE_DOWN_PAYMENT': ['max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'],  # Jours depuis la décision d'octroi
    'CNT_PAYMENT': ['max', 'mean'],           # Nombre d'échéances
    'DAYS_TERMINATION': ['max'],              # Date de fin du contrat
    'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
    'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
    'DOWN_PAYMENT_TO_CREDIT': ['mean'],
}

#  Agrégations sur les prêts actifs en cours 
PREVIOUS_ACTIVE_AGG = {
    'SK_ID_PREV': ['nunique'],
    'SIMPLE_INTERESTS': ['mean'],         # Taux d'intérêt simplifié
    'AMT_ANNUITY': ['max', 'sum'],
    'AMT_APPLICATION': ['max', 'mean'],
    'AMT_CREDIT': ['sum'],
    'AMT_DOWN_PAYMENT': ['max', 'mean'],
    'DAYS_DECISION': ['min', 'mean'],
    'CNT_PAYMENT': ['mean', 'sum'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    'AMT_PAYMENT': ['sum'],               # Total déjà remboursé
    'INSTALMENT_PAYMENT_DIFF': ['mean', 'max'],  # Retard cumulé
    'REMAINING_DEBT': ['max', 'mean', 'sum'],    # Dette restante
    'REPAYMENT_RATIO': ['mean'],          # Taux de remboursement
}

#  Agrégations sur les demandes approuvées 
PREVIOUS_APPROVED_AGG = {
    'SK_ID_PREV': ['nunique'],
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_CREDIT': ['min', 'max', 'mean'],
    'AMT_DOWN_PAYMENT': ['max'],
    'AMT_GOODS_PRICE': ['max'],
    'HOUR_APPR_PROCESS_START': ['min', 'max'],
    'DAYS_DECISION': ['min', 'mean'],
    'CNT_PAYMENT': ['max', 'mean'],
    'DAYS_TERMINATION': ['mean'],
    'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
    'APPLICATION_CREDIT_DIFF': ['max'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
    'DAYS_FIRST_DRAWING': ['max', 'mean'],  # Date du premier versement
    'DAYS_FIRST_DUE': ['min', 'mean'],      # Date de la première échéance
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    'DAYS_LAST_DUE': ['max', 'mean'],
    'DAYS_LAST_DUE_DIFF': ['min', 'max', 'mean'],  # Écart entre date de fin prévue et réelle
    'SIMPLE_INTERESTS': ['min', 'max', 'mean'],
}

#  Agrégations sur les demandes refusées 
PREVIOUS_REFUSED_AGG = {
    'AMT_APPLICATION': ['max', 'mean'],
    'AMT_CREDIT': ['min', 'max'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['max', 'mean'],
    'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean', 'var'],
    'APPLICATION_CREDIT_RATIO': ['min', 'mean'],
    'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
    'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
    'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
}

#  Agrégations sur les prêts avec retards de paiement 
PREVIOUS_LATE_PAYMENTS_AGG = {
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    'APPLICATION_CREDIT_DIFF': ['min'],
    'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
    'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
    'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
}

#  Agrégations par type de prêt précédent (consommation / cash) 
PREVIOUS_LOAN_TYPE_AGG = {
    'AMT_CREDIT': ['sum'],
    'AMT_ANNUITY': ['mean', 'max'],
    'SIMPLE_INTERESTS': ['min', 'mean', 'max', 'var'],
    'APPLICATION_CREDIT_DIFF': ['min', 'var'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
    'DAYS_DECISION': ['max'],
    'DAYS_LAST_DUE_1ST_VERSION': ['max', 'mean'],
    'CNT_PAYMENT': ['mean'],
}

#  Agrégations sur les demandes récentes (12 ou 24 derniers mois) 
PREVIOUS_TIME_AGG = {
    'AMT_CREDIT': ['sum'],
    'AMT_ANNUITY': ['mean', 'max'],
    'SIMPLE_INTERESTS': ['mean', 'max'],
    'DAYS_DECISION': ['min', 'mean'],
    'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
    'APPLICATION_CREDIT_DIFF': ['min'],
    'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
    'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
    'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
    'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
}

#  Agrégations sur les soldes POS/Cash 
POS_CASH_AGG = {
    'SK_ID_PREV': ['nunique'],              # Nombre de prêts POS distincts
    'MONTHS_BALANCE': ['min', 'max', 'size'],  # Durée de l'historique
    'SK_DPD': ['max', 'mean', 'sum', 'var'],   # Jours de retard (Days Past Due)
    'SK_DPD_DEF': ['max', 'mean', 'sum'],      # Jours de retard définis (seuil institution)
    'LATE_PAYMENT': ['mean']               # Taux de mois avec retard
}

#  Agrégations sur les paiements d'échéances 
INSTALLMENTS_AGG = {
    'SK_ID_PREV': ['size', 'nunique'],         # Nombre total de lignes et prêts distincts
    'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],  # Dates de paiement
    'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],  # Montants des échéances dues
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],     # Montants effectivement payés
    'DPD': ['max', 'mean', 'var'],             # Retard en jours (Days Past Due)
    'DBD': ['max', 'mean', 'var'],             # Avance en jours (Days Before Due)
    'PAYMENT_DIFFERENCE': ['mean'],            # Solde non payé par échéance
    'PAYMENT_RATIO': ['mean'],                 # Taux de couverture de l'échéance
    'LATE_PAYMENT': ['mean', 'sum'],           # Taux et nombre de paiements tardifs
    'SIGNIFICANT_LATE_PAYMENT': ['mean', 'sum'],  # Retards significatifs (>5%)
    'LATE_PAYMENT_RATIO': ['mean'],            # Ratio moyen des paiements tardifs
    'DPD_7': ['mean'],                         # Taux de retards >= 7 jours
    'DPD_15': ['mean'],                        # Taux de retards >= 15 jours
    'PAID_OVER': ['mean']                      # Taux de surpaiements
}

#  Agrégations temporelles des paiements (périodes récentes) 
INSTALLMENTS_TIME_AGG = {
    'SK_ID_PREV': ['size'],
    'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
    'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
    'DPD': ['max', 'mean', 'var'],
    'DBD': ['max', 'mean', 'var'],
    'PAYMENT_DIFFERENCE': ['mean'],
    'PAYMENT_RATIO': ['mean'],
    'LATE_PAYMENT': ['mean'],
    'SIGNIFICANT_LATE_PAYMENT': ['mean'],
    'LATE_PAYMENT_RATIO': ['mean'],
    'DPD_7': ['mean'],
    'DPD_15': ['mean'],
}

#  Agrégations sur les soldes de cartes de crédit 
CREDIT_CARD_AGG = {
    'MONTHS_BALANCE': ['min'],                    # Ancienneté de l'historique
    'AMT_BALANCE': ['max'],                       # Solde maximal
    'AMT_CREDIT_LIMIT_ACTUAL': ['max'],           # Plafond de crédit maximal
    'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],   # Retraits ATM
    'AMT_DRAWINGS_CURRENT': ['max', 'sum'],       # Total des retraits
    'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],   # Retraits en point de vente
    'AMT_INST_MIN_REGULARITY': ['max', 'mean'],   # Paiement minimum régulier
    'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'sum', 'var'],  # Paiements effectués
    'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],      # Total des créances
    'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],  # Nombre de retraits ATM
    'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],      # Nombre total de retraits
    'CNT_DRAWINGS_POS_CURRENT': ['mean'],                # Retraits en point de vente
    'SK_DPD': ['mean', 'max', 'sum'],             # Jours de retard
    'SK_DPD_DEF': ['max', 'sum'],                 # Jours de retard définis
    'LIMIT_USE': ['max', 'mean'],                 # Taux d'utilisation du plafond
    'PAYMENT_DIV_MIN': ['min', 'mean'],           # Ratio paiement / minimum dû
    'LATE_PAYMENT': ['max', 'sum'],               # Retards de paiement
}

#  Agrégations temporelles des cartes de crédit (12, 24, 48 derniers mois)
CREDIT_CARD_TIME_AGG = {
    'CNT_DRAWINGS_ATM_CURRENT': ['mean'],  # Fréquence des retraits ATM récents
    'SK_DPD': ['max', 'sum'],              # Retards récents
    'AMT_BALANCE': ['mean', 'max'],        # Solde récent
    'LIMIT_USE': ['max', 'mean']           # Utilisation récente du plafond
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prétraitement des données Home Credit")
    parser.add_argument("--debug", action="store_true", help="Mode debug : charge 30 000 lignes seulement")
    parser.add_argument("--output", type=str, default="data/preprocessed.csv", help="Chemin du CSV de sortie")
    args = parser.parse_args()

    with timer("Prétraitement complet"):
        df = load_dataset(debug=args.debug)

    output = args.output
    if os.path.isdir(output) or output.endswith("/") or output.endswith("\\"):
        output = os.path.join(output, "preprocessed.csv")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    print(f"Sauvegarde du DataFrame ({df.shape[0]} lignes, {df.shape[1]} colonnes) dans '{output}'...")
    df.to_csv(output, index=False)
    print(f"Fichier sauvegardé : {output}")