import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline 

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB

MASS_FILENAME = 'mass.csv'
T6_MANEUVER_GRADE_FILENAME = './data/t6_maneuver_grades.csv'
T38_MANEUVER_GRADE_FILENAME = './data/t38_maneuver_grades.csv'
IFF_MANEUVER_GRADE_FILENAME = './data/iff_maneuver_grades.csv'

def parse_mass(mass_df: pd.DataFrame, airframe: str):
    grouping_df = mass_df.loc[np.where(mass_df['NAME_NM'].str.contains(airframe) & mass_df['MERIT_RANK_SYS_CATG_NAME'].isin(['Daily Maneuver T-Score', 'Academics T-Score', 'Category Check Maneuver T-Score']))]
    grouping_df = grouping_df[['BASE_RSRC_ID', 'MERIT_RANK_SYS_CATG_NAME', 'RAW_SCORE_DV']]
    grouping_df = pd.pivot(grouping_df, index='BASE_RSRC_ID', columns=['MERIT_RANK_SYS_CATG_NAME'], values='RAW_SCORE_DV')
    grouping_df['Computed'] = grouping_df['Academics T-Score']*.1 + grouping_df['Daily Maneuver T-Score']*.2 + grouping_df['Category Check Maneuver T-Score']*.4
    return grouping_df

def t6_replace(grades_df: pd.DataFrame) -> pd.DataFrame:
    output_grades_df = grades_df.copy() 

    grade_mapping = {
        'AREA ORIENTATION (LEAD)': 'AREA ORIENTATION',
        'ASR FINAL': 'ASR/PAR FINAL',
        'BASIC/ENROUTE AIRCRAFT CONTROL': 'BASIC AIRCRAFT CONTROL',
        'CIRCLING APPROACH FINAL': 'CIRCLING',
        'CIRCLING APPROACH': 'CIRCLING',
        'CLEARING VISUAL LOOKOUT': 'CLEARING',
        'CLEARING VISUAL / LOOKOUT': 'CLEARING',
        'CLIMB & DEPARTURE (LEAD)': 'DEPARTURE',
        'CLIMB & DEPARTURE': 'DEPARTURE',
        'CLOSED PATTERN': 'CLOSED TRAFFIC',
        'COMMUNICATION & VISUAL SIGNALS': 'COMMUNICATION',
        'COMPOSITE CROSS CHECK': 'CROSS-CHECK',
        'COMPOSITE CROSS-CHECK': 'CROSS-CHECK',
        'CROSSUNDER (WING,': 'CROSSUNDER',
        'ECHELON TURNS': 'ECHELON TURN (WING)',
        'ENROUTE DESCENT/LETDOWN': 'ENROUTE DESCENT',
        'ENROUTE DESCENT/LETDOWN RADAR': 'ENROUTE DESCENT',
        'FIGHTING WING (WING)': 'FIGHTING WING',
        'FIGHTING WING/WEDGE': 'FIGHTING WING',
        'FINGERTIP (WING)': 'FINGERTIP',
        'G-AWARENESS/EXERCISE': 'G-AWARENESS EXERCISE',
        'GPS FINAL': 'GPS APPROACH',
        'HIGH ALTITUDE POWER LOSS (HAPL)': 'HIGH-ALTITUDE POWER LOSS (HAPL)',
        'ILS FINAL': 'ILS APPROACH',
        'INFLIGHT CHECKS/COCKPIT COORDINATION': 'INFLIGHT CHECKS',
        'INFLIGHT COMPUTATIONS': 'INFLIGHT PLANNING',
        'INFLIGHT PLAN/AREA ORIENTATIONS': 'INFLIGHT PLANNING',
        'INFLIGHT PLANNING (LEAD)': 'INFLIGHT PLANNING',
        'INFLIGHT PLANNING/AREA ORIENTATION': 'INFLIGHT PLANNING',
        'INSTRUMENT APPROACH (FULL PROCEDURES)': 'INSTRUMENT APPROACH',
        'INSTRUMENT APPROACH (RADAR VECTORS)': 'INSTRUMENT APPROACH',
        'INSTRUMENT APPROACH (VECTORS/FULL PROC)': 'INSTRUMENT APPROACH',
        'INSTRUMENT APPROACH PROCEDURES': 'INSTRUMENT APPROACH',
        'LETDOWN/TRAFFIC ENTRY (LEAD)': 'LETDOWN/TRAFFIC ENTRY',
        'LEVEL OFF': 'LEVEL-OFF',
        'LOC FINAL': 'LOCALIZER APPROACH',
        'LOCALIZER FINAL': 'LOCALIZER APPROACH',
        'NO FLAP LANDING': 'NO-FLAP LANDING',
        'NO-FLAP PATTERN': 'NO-FLAP OVERHEAD PATTERN',
        'NORMAL PATTERN': 'NORMAL OVERHEAD PATTERN',
        'OVERHEAD PATTERN': 'NORMAL OVERHEAD PATTERN',
        'PAR FINAL': 'PAR APPROACH',
        'REJOINS (TAC OR STANDARD)': 'REJOINS',
        'ROUTE (WING)': 'ROUTE',
        'SINGLE SHIP INSTRUMENT APPR PROCEDURES': 'INSTRUMENT APPROACH',
        'STRAIGHT-AHEAD REJOIN (WING)': 'STRAIGHT-AHEAD REJOIN',
        'TACTICAL (LEAD)': 'TACTICAL MANEUVERING (LEAD)',
        'TACTICAL REJOIN-STRAIGHT AHEAD (WING)': 'TACTICAL STRAIGHT-AHEAD REJOIN',
        'TAKEOFF/TRANSITION TO INSTRUMENTS': 'TAKEOFF',
        'TRAFFIC PATTERN STALLS (NORM OR NF)': 'TRAFFIC PATTERN STALLS',
        'TRANSITION TO LAND/LANDING FROM APPROACH': 'TRANSITION TO LANDING',
        'TRANSITION TO LDG/LDG FROM AN APPROACH': 'TRANSITION TO LANDING',
        'UNUSUAL ATTITUDES': 'UNUSUAL ATTITUDE RECOVERIES',
        'VOR FINAL APPROACH': 'VOR APPROACH',
        'VOR FINAL': 'VOR APPROACH',
    }

    output_grades_df['SYL_EVNT_ITM_NAME_NM'] = output_grades_df['SYL_EVNT_ITM_NAME_NM'].map(lambda x: x if (x not in grade_mapping) else grade_mapping[x])

    return output_grades_df

def t38_replace(grades_df: pd.DataFrame) -> pd.DataFrame:
    output_grades_df = grades_df.copy()
    
    grade_mapping = {
        'CIRCLING APPROACH': 'CIRCLING',
        'G AWARENESS': 'G-AWARENESS',
        'HIGH/LOW ALTITUDE APPROACH': 'HIGH/LOW ALTITUDE APPROACHES',
        'GO AROUND': 'GO-AROUND',
        'HUD OFF NO-FLAP OVERHEAD': 'HUD OFF NO-FLAP OVERHEAD PATTERN',
        'HUD OFF NORMAL VFR STRAIGHT-IN': 'HUD OFF NORMAL STRAIGHT-IN',
        'INSTRUMENT APPROACH - NO FLAP': 'INSTRUMENT APPROACH - NO-FLAP',
        'LEVEL OFF': 'LEVEL-OFF',
        'NO-FLAP STRAIGHT-IN PATTERN': 'NO-FLAP STRAIGHT-IN',
        'NON-PRECISION INSTR APPROACH - SSE': 'NON-PRECISION INSTRUMENT - SSE',
        'PRECISION INSTRUMENT APPROACH - SSE': 'PRECISION INSTRUMENT - SSE',
        'RNAV APPROACH': 'RNAV (GPS) APPROACH',
        'SITUATIONAL AWARENESS (AIRMANSHIP)': 'SITUATIONAL AWARENESS',
        '"SPLIT ""S"""': 'SPLIT S',
        'STRAIGHT AHEAD REJOIN': 'STRAIGHT-AHEAD REJOIN',
        'TACTICAL POSITION': 'TACTICAL POSITION (WING)',
        'TACTICAL STRAIGHT AHEAD REJOIN': 'TACTICAL STRAIGHT-AHEAD REJOIN',
        'TACTICAL TURNING REJOIN': 'TACTICAL REJOIN',
        'TURNING REJOIN #3 / #4': 'TURNING REJOIN 3/4',
        'TURNING REJOIN -- #3': 'TURNING REJOIN 3/4',
        'TURNING REJOIN-2': 'TURNING REJOIN 2',
        'TURNING REJOIN -- #2': 'TURNING REJOIN 2',
        'UNUSUAL ATTITUDE RECOVERY': 'UNUSUAL ATTITUDE RECOVERIES',
        'VOR/DME OR TACAN APPROACH': 'VOR/DME/TACAN APPROACH',
    }

    output_grades_df['SYL_EVNT_ITM_NAME_NM'] = output_grades_df['SYL_EVNT_ITM_NAME_NM'].map(lambda x: x if (x not in grade_mapping) else grade_mapping[x])

    return output_grades_df

def get_grade_and_pass_rate_for_event(grades_df: pd.DataFrame, event: str) -> pd.DataFrame:
    ride_maneuvers_df = grades_df.loc[(grades_df['SYL_EVNT_NAME_NM'] == event) & (grades_df['SYL_EVNT_INDEX_IN'] == 1)]
    ride_grades_df = ride_maneuvers_df.groupby(['BASE_RSRC_ID']).mean(numeric_only=True)
    ride_grades_df = ride_grades_df[['OVRAL_GRADE_QY', 'ITEM_GRADE_QY']]
    ride_grades_df['OVRAL_GRADE_QY'] = np.where(ride_grades_df['OVRAL_GRADE_QY'] >= 4.0, 1, 0)
    ride_grades_df.rename(columns={'OVRAL_GRADE_QY': event + ' PASS', 'ITEM_GRADE_QY': event + '_ITEM_GRADE'}, inplace=True)
    return ride_grades_df


def compute_gradesheet_averages(grades_df: pd.DataFrame, airframe: str):
    grades_wo_ng_df = grades_df[grades_df.ITEM_GRADE_LABEL != 'NG']
    #item_grades_df = grades_wo_ng_df[["BASE_RSRC_ID", "SYL_EVNT_ITM_NAME_NM", "ITEM_GRADE_QY"]]

    replaced_grades_df = grades_wo_ng_df # item_grades_df

    if (airframe == 'T6') or (airframe == 'T-6'):
        replaced_grades_df = t6_replace(grades_wo_ng_df)
    elif (airframe == 'T38') or (airframe == 'T-38'):
        replaced_grades_df = t38_replace(grades_wo_ng_df)

    avg_grades_per_student_df = replaced_grades_df.groupby(["BASE_RSRC_ID", "SYL_EVNT_ITM_NAME_NM"], as_index=False).mean(numeric_only=True)
    avg_grades_per_student_df = avg_grades_per_student_df.pivot(index="BASE_RSRC_ID", columns="SYL_EVNT_ITM_NAME_NM", values="ITEM_GRADE_QY")
    avg_grades_per_student_df.index.names = ['BASE_RSRC_ID']

    # get form checkride scores (F4390)
    if (airframe == 'T6') or (airframe == 'T-6'):
        for event in ['F4390', 'I4390']:
            grades_for_event_df = get_grade_and_pass_rate_for_event(grades_wo_ng_df, event)
            avg_grades_per_student_df = pd.merge(avg_grades_per_student_df, grades_for_event_df, how='inner', on='BASE_RSRC_ID')

    perc = 80.0
    min_count = int((perc/100) * avg_grades_per_student_df.shape[0] + 1)
    avg_grades_per_student_df = avg_grades_per_student_df.dropna(axis= 1, thresh=min_count)

    return avg_grades_per_student_df
    # calculate fairs and unsats
    #overall_grades_df = grades_wo_ng_df[["BASE_RSRC_ID", "SYL_EVNT_NAME_NM", "SYL_EVNT_INDEX_IN", "GRADE_NAME_NM"]]
    #unsat_grades_df = overall_grades_df.loc[(overall_grades_df['GRADE_NAME_NM'] == 'U') | (overall_grades_df['GRADE_NAME_NM'] == 'F')]
    #unsat_grade_sum_df = unsat_grades_df.groupby(['BASE_RSRC_ID', 'SYL_EVNT_NAME_NM', 'SYL_EVNT_INDEX_IN']).size().groupby('BASE_RSRC_ID').count()
    #unsat_grade_sum_df = unsat_grade_sum_df.rename('NumUnsatsFairs')

    #output_df = pd.merge(avg_grades_per_student_df, unsat_grade_sum_df, how='left', on='BASE_RSRC_ID')
    #output_df['NumUnsatsFairs'] = output_df['NumUnsatsFairs'].fillna(0.0)

    ## calculate 88s
    #evt_modifier_grades_df = grades_wo_ng_df[["BASE_RSRC_ID", "SYL_EVNT_NAME_NM", "SYL_EVNT_INDEX_IN", "EVNT_MODIFIER_CD"]]
    #print(evt_modifier_grades_df)
    #ipc_grades_df = evt_modifier_grades_df.loc[evt_modifier_grades_df['EVNT_MODIFIER_CD'] == '88'] 
    #print(ipc_grades_df)
    #ipc_grade_sum_df = ipc_grades_df.groupby(['BASE_RSRC_ID', 'SYL_EVNT_NAME_NM', 'SYL_EVNT_INDEX_IN']).size().groupby('BASE_RSRC_ID').count()
    #ipc_grade_sum_df = ipc_grade_sum_df.rename('NumIPCs')

    #output_df = pd.merge(output_df, ipc_grade_sum_df, how='left', on='BASE_RSRC_ID')
    #output_df['NumIPCs'] = output_df['NumIPCs'].fillna(0.0)
    #print(output_df)
    #exit()
    #return output_df 


def compute_mass(grades_df: pd.DataFrame, airframe: str):
    grades_wo_ng_df = grades_df[grades_df.ITEM_GRADE_LABEL != 'NG']
    grades_wo_ng_df = grades_wo_ng_df[["BASE_RSRC_ID", "SYL_EVNT_ITM_NAME_NM", "ITEM_GRADE_QY"]]
    avg_grades_per_student_df = grades_wo_ng_df.groupby(["BASE_RSRC_ID"]).mean(numeric_only=True)
    avg_grades_per_student_df = avg_grades_per_student_df.rename(columns={'ITEM_GRADE_QY': airframe + ' MASS'})
    
    return avg_grades_per_student_df

def get_features_selected(clf) -> any:
    trans_dict = dict(clf.best_estimator_['features'].transformer_list)
    mask = trans_dict['select'].get_support()
    features_selected = clf.best_estimator_['preprocessor'].get_feature_names_out()

    return features_selected[mask]


def fit_score_print(preprocessor, classifier, params, x_train, y_train, x_test, y_test):
    pca = PCA(n_components=2)
    selector = SelectKBest(k='all')
    combined_features = FeatureUnion({("pca", pca), ("select", selector)})

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("features", combined_features),
            ("classifier", classifier)
        ]
    )

    clf = GridSearchCV(pipeline, param_grid=params, cv=5)
    clf.fit(x_train, y_train)
    features_selected = get_features_selected(clf)

    print(clf.best_score_)
    print(clf.best_params_)
    print(clf.classes_)
    print(features_selected)

    predictions = clf.predict(x_test)
    probs = clf.predict_proba(x_test)
    probs = [prob_list[1] for prob_list in probs]

    pred_probs_df = pd.DataFrame({'predictions': predictions, 'probabilities': probs})

    return (clf.best_score_, pred_probs_df)

def remove_extraneous_duplicated_students(grades_df: pd.DataFrame) -> pd.DataFrame:
    students_df = grades_df[['BASE_RSRC_ID', 'STUDENT_NAME_NM', 'NAME_NM']]
    students_df = students_df.set_index('BASE_RSRC_ID')
    students_df = students_df[~students_df.index.isin([1160023198, 1160037136, 1160020542, 1160020530, 1160023199, 1160023199, 1160037236])]
    students_df = students_df[~students_df.index.duplicated(keep='first')]

    grades_df = grades_df[grades_df['BASE_RSRC_ID'].isin(students_df.index)]

    return grades_df

def read_in_maneuver_grades(airframe: str) -> pd.DataFrame:
    grades_df = pd.read_csv('data/' + airframe.lower() + '_maneuver_grades.csv')
    grades_df = remove_extraneous_duplicated_students(grades_df)
    return grades_df

def get_averages(airframe: str, grades_df: pd.DataFrame) -> pd.DataFrame:
    CACHE_FOLDER = 'cache'
    CACHE_FILEPATH = CACHE_FOLDER + '/' + airframe.lower() + '_averages.csv'

    if (not os.path.exists(CACHE_FOLDER)):
        os.makedirs(CACHE_FOLDER)

    averages_file = Path(CACHE_FILEPATH)
    averages_df = pd.DataFrame()

    if (not averages_file.exists()):
        averages_df = compute_gradesheet_averages(grades_df, airframe)
        averages_df.to_csv(CACHE_FILEPATH)
    else:
        averages_df = pd.read_csv(CACHE_FILEPATH, index_col='BASE_RSRC_ID')

    return averages_df

def get_mass(airframe: str, grades_df: pd.DataFrame) -> pd.DataFrame:
    CACHE_FOLDER = 'cache'
    CACHE_FILEPATH = CACHE_FOLDER + '/' + airframe.lower() + '_mass.csv'

    if (not os.path.exists(CACHE_FOLDER)):
        os.makedirs(CACHE_FOLDER)

    averages_file = Path(CACHE_FILEPATH)
    mass_df = pd.DataFrame()

    if (not averages_file.exists()):
        mass_df = compute_mass(grades_df, airframe)
        mass_df.to_csv(CACHE_FILEPATH)
    else:
        mass_df = pd.read_csv(CACHE_FILEPATH)

    return mass_df

def get_weighted_correlations(t38_averages_df: pd.DataFrame, iff_mass_df: pd.DataFrame) -> pd.DataFrame:
    all_mass_df = pd.merge(t38_averages_df, iff_mass_df, how='inner', on='BASE_RSRC_ID')
    t38_iff_corrs = all_mass_df.corr()['IFF MASS']
    t38_iff_corrs = t38_iff_corrs.drop('IFF MASS')
    total_weight = t38_iff_corrs.sum()
    t38_iff_corr_weights = t38_iff_corrs / total_weight

    return t38_iff_corr_weights

def get_weighted_mass(airframe: str, averages_df: pd.DataFrame, mass_df: pd.DataFrame) -> pd.DataFrame:
    corr_weights_df = get_weighted_correlations(averages_df, mass_df)

    student_weighted_averages = averages_df * corr_weights_df
    student_weighted_averages = student_weighted_averages.sum(axis=1)
    weighted_mass_df = student_weighted_averages
    weighted_mass_df = weighted_mass_df.rename(airframe.upper() + ' MASS')

    return weighted_mass_df

def run_track_selector():
    print("enter run")

    t6_maneuver_grades_df = read_in_maneuver_grades('T6')
    t38_maneuver_grades_df = read_in_maneuver_grades('T38')
    iff_maneuver_grades_df = read_in_maneuver_grades('IFF')

    t6_averages_df = get_averages('T6', t6_maneuver_grades_df)
    t38_averages_df = get_averages('T38', t38_maneuver_grades_df)

    t38_mass_df = get_mass('T38', t38_maneuver_grades_df)
    iff_mass_df = get_mass('IFF', iff_maneuver_grades_df)

    print(t6_averages_df)
    print(t38_averages_df)
    print(t38_mass_df)
    print(iff_mass_df)

    tims_mass_df = pd.read_csv('data/mass_data.csv')
    academics_df = tims_mass_df[['BASE_RSRC_ID', 'NAME_NM', 'SYL_OVRAL_ST_NAME_NM', 'OVRAL_ST_EFF_DATE_TIME_DT', 'MERIT_RANK_SYS_CATG_NAME','RAW_SCORE_DV']]
    academics_df = academics_df[(academics_df['MERIT_RANK_SYS_CATG_NAME'] == 'Academics T-Score') & (academics_df['SYL_OVRAL_ST_NAME_NM'] == 'Complete') & (academics_df['NAME_NM'].str.contains('T-6'))]
    academics_df = academics_df[['BASE_RSRC_ID', 'RAW_SCORE_DV']]
    academics_df = academics_df.set_index('BASE_RSRC_ID')
    academics_df = academics_df.rename(columns={'RAW_SCORE_DV': 'T6 ACADEMICS'})
    print(academics_df)
    t6_averages_df = pd.merge(t6_averages_df, academics_df, how='inner', on='BASE_RSRC_ID')
    print(t6_averages_df)

   # t6_mass_df = get_mass('T6', t6_maneuver_grades_df)
   # combined_df = pd.merge(t6_mass_df, t38_mass_df, how='inner', on='BASE_RSRC_ID')
   # print(combined_df)
   # combined_df.plot(kind='scatter', x='T6 MASS', y='T38 MASS')
   # plt.show()
   # exit()

    t38_mass_df = get_weighted_mass('T38', t38_averages_df, iff_mass_df)
    merged_df = pd.DataFrame.merge(t6_averages_df, t38_mass_df, on='BASE_RSRC_ID') #may delete left
    merged_df = merged_df.sort_values('T38 MASS', ascending=False).reset_index()
    corr_df = merged_df.corr()[['T38 MASS']]
    corr_df = corr_df.sort_values('T38 MASS', ascending=False)
    corr_df.to_csv('corr.csv')
    exit()

    perc = 50.0
    merged_df['Class'] = np.where(merged_df.index <= ((perc/100)*len(merged_df.index)), 1, 0)

    merged_df = merged_df.sort_values('T38 MASS', ascending=False)
    x_df = merged_df.drop(columns=['T38 MASS', 'BASE_RSRC_ID', 'Class'])
    y_df = merged_df['Class']

    x_train, X_test, y_train, Y_test = train_test_split(x_df, y_df, test_size=.2)

    categorical_features = ['F4390 PASS', 'I4390 PASS']
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    numeric_features = x_df.columns.drop(categorical_features)
    numeric_transformer = Pipeline(
        steps=[("imputer", KNNImputer()), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features)
        ]
    )

    params = {
        #"features__pca__n_components": np.arange(1, 3),
        #"features__select__score_func": [f_classif],
        "features__select__k": np.arange(1, 8)
    }

    log_params = params.copy()
    log_params['classifier__C'] = np.arange(.1, 1.1, .2)

    svc_params = log_params.copy()
    svc_params['classifier__gamma'] = ['auto', 'scale']

    classifiers = [
        #(SVC(), svc_params),
        #(LinearSVC(max_iter=100000), log_params),
        (LogisticRegression(), log_params),
        #(RandomForestClassifier(), params),
        (KNeighborsClassifier(), params),
        #(DecisionTreeClassifier(),params),
        (GaussianNB(), params),
        #(AdaBoostClassifier(), params),
        #(GaussianProcessClassifier(1.0 * RBF(1.0)), params),
        #(MLPClassifier(max_iter=5000), params)
    ]

    all_predictions = pd.DataFrame()
    all_probabilities = pd.DataFrame()

    for (classifier, params) in classifiers:
        (score, prediction_probability_df) = fit_score_print(preprocessor, classifier, params, x_train, y_train, X_test, Y_test)
        prediction_probability_df.to_csv('pred_probs.csv')
        all_predictions[classifier.__class__.__name__] = prediction_probability_df['predictions']
        all_probabilities[classifier.__class__.__name__] = prediction_probability_df['probabilities']
    
    all_predictions['Average'] = all_predictions.median(axis = 1).round()
    all_probabilities['Average'] = all_probabilities.mean(axis = 1)
   # print(all_predictions)
   # probs = all_predictions['Average']
   # sorted_probs = probs.copy()
   # print(sorted_probs)
   # sorted_probs = sorted_probs.sort_values(ascending=False)
   # print(sorted_probs)
   # min_prob_to_be_above = sorted_probs[int(len(sorted_probs) * .15)]
   # print(min_prob_to_be_above)
   # predictions = [1 if val >= min_prob_to_be_above else 0 for val in probs]
   # all_predictions['Average'] = predictions
    print(all_predictions)
    all_predictions.to_csv('all_predictions.csv')
    all_probabilities.to_csv('all_probabilities.csv')

    comparison_df = pd.DataFrame()
    comparison_df["Predictions"] = all_predictions['Average'].to_numpy()
    comparison_df['Actual'] = Y_test.to_numpy()

    comparison_df["Equals"] = comparison_df.apply(lambda x : True if x['Predictions'] == x['Actual'] else False, axis=1)
    comparison_df['Probability'] = all_probabilities['Average']
    comparison_df = comparison_df.sort_values('Probability', ascending=False)
    comparison_df.to_csv('comparison.csv')
    correct_count = (comparison_df['Equals']).sum()
    total_count = len(comparison_df.index)
    accuracy = (correct_count / total_count) * 100

    print(correct_count)
    print(total_count)

    print(f"Complete model accuracy {accuracy}%")

    print("exit run")    

def get_ids_for(grades_df: pd.DataFrame, filter_df: pd.DataFrame) -> pd.DataFrame:
    grades_df['NAME_NM'] = grades_df['NAME_NM'].fillna("")
    personnel_df = grades_df.groupby(['BASE_RSRC_ID', 'STUDENT_NAME_NM', 'NAME_NM'], as_index=False).count()[['BASE_RSRC_ID', 'STUDENT_NAME_NM', 'NAME_NM']]
    personnel_df = personnel_df.set_index('BASE_RSRC_ID')
    filtered_personnel_df = personnel_df.reset_index().merge(filter_df, how='inner', on=['STUDENT_NAME_NM', 'NAME_NM']).set_index('BASE_RSRC_ID')
    return filtered_personnel_df

def get_maneuver_grades_for(maneuver_grades_df: pd.DataFrame, personnel: pd.DataFrame) -> pd.DataFrame:
    grades_for_personnel = maneuver_grades_df.loc[maneuver_grades_df['BASE_RSRC_ID'].isin(personnel.index)]
    return grades_for_personnel

def get_maneuver_grades_excluding(maneuver_grades_df: pd.DataFrame, personnel: pd.DataFrame) -> pd.DataFrame:
    grades_for_personnel = maneuver_grades_df.loc[~maneuver_grades_df['BASE_RSRC_ID'].isin(personnel.index)]
    return grades_for_personnel

def remove_fake_students(grades_df: pd.DataFrame) -> pd.DataFrame:
    fake_profiles = ['', 'IFF-A, Dummy', 'Wyatt, Lisa', 'IFF-B, Dummy']
    grades_df = grades_df[~grades_df['STUDENT_NAME_NM'].isin(fake_profiles)]
    return grades_df

def remove_no_grade_items(grades_df: pd.DataFrame) -> pd.DataFrame:
    return grades_df.loc[grades_df['ITEM_GRADE_LABEL'] != 'NG']

def remap_iff_sorties(grades_df: pd.DataFrame) -> pd.DataFrame:
    sortie_mapping = {
        'ACM-1B': 'ACM-1',
        'ACM-2B': 'ACM-2',
        'CAS-1B': 'CAS-1',
        'CAS-1BAF': 'CAS-1',
        'CAS-1C': 'CAS-1',
        'DB-1C': 'DB-1',
        'DB-2A': 'DB-2',
        'DB-2B': 'DB-2',
        'DB-2C': 'DB-2',
        'F-1A': 'F-1',
        'F-1B': 'F-1',
        'F-1C': 'F-1',
        'F-3C': 'F-3',
        'H-1ABAF': 'H-1',
        'H-1C': 'H-1',
        'H-1CF': 'H-1',
        'HB-1B': 'HB-1',
        'HB-2B': 'HB-2',
        'OB-1C': 'OB-1',
        'OB-2C': 'OB-2',
        'S-1C': 'S-1',
        'S-1CF': 'S-1',
        'S-2C': 'S-2',
        'S-2CF': 'S-2',
        'S-3C': 'S-3',
        'S-3CF': 'S-3',
        'S-4BAF': 'S-4',
        'S-4C': 'S-4',
        'S-4CF': 'S-4',
        'SAT-1BF': 'SAT-1',
        'SAT-1BA': 'SAT-1',
        'SAT-1C': 'SAT-1',
        'SAT-2C': 'SAT-2',
    }

    grades_df['SYL_EVNT_NAME_NM'] = grades_df['SYL_EVNT_NAME_NM'].map(lambda x: x if (x not in sortie_mapping) else sortie_mapping[x])
    return grades_df

def remap_iff_items(grades_df: pd.DataFrame) -> pd.DataFrame:
    item_mapping = {
        'BRK TURN EX/ACCELERATED STALL': 'BREAK TURN/ACCELERATED STALL',
        'ECHELON FORMATION': 'ECHELON',
        'FIGHT ANALYSIS (HBFM)': 'FIGHT ANALYSIS',
        'FIGHT ANALYSIS (OBFM)': 'FIGHT ANALYSIS',
        'FINGERTIP FORMATION': 'FINGERTIP',
        'FORMATION APPROACH AND LANDING': 'FORMATION APPROACH (WING)',
        'FORMATION APPROACH AND LANDING (WING)': 'FORMATION APPROACH (WING)',
        'FOUR(THREE)-SHIP FORMATION - BASIC': 'FOUR-SHIP FORMATION - BASIC',
        'FOUR(THREE)-SHIP FORMATION - TACTICAL': 'FOUR-SHIP FORMATION - TACTICAL',
        'HIGH/LOW SPEED DIVE RECOVERY': 'HIGH/LOW-SPEED DIVE RECOVERY',
        'LEVEL/POPUP RANGE PROCEDURES/PATTERNS': 'LEVEL/POPUP RANGE',
        'NO-FLAP APPROACH/LANDING': 'NO FLAP APPROACH/LANDING',
        'POSITION/TURNS': 'TWO-SHIP FORMATION -- TACTICAL',
        'QUARTER PLANE EXERCISE': 'QUARTER-PLANE EXERCISE',
        'REJOIN': 'REJOINS',
        'ROUTE FORMATION': 'ROUTE',
        'REVERSAL/SCISSORS EXERCISE': 'REVERSAL / SCISSORS EXERCISE',
        'SAFE-ESCAPE MANEUVER': 'SAFE ESCAPE MANEUVER',
        'SIMULATED GUN SHOT': 'SIMULATED GUNSHOT',
        'TACTICAL REJOIN': 'REJOINS',
        'OFF / DEF WEZ RECOGNITION': 'WEZ RECOGNITION',
        'WEAPONS EMPLOYMENT (SA)': 'WEAPONS EMPLOYMENT',
        'WEZ RECOGNITION (HBFM)': 'WEZ RECOGNITION',
    }

    grades_df['SYL_EVNT_ITM_NAME_NM'] = grades_df['SYL_EVNT_ITM_NAME_NM'].map(lambda x: x if (x not in item_mapping) else item_mapping[x])
    return grades_df

def run_upt_iff_analysis() -> None:
    # read in IFF grades after a 6/18/2019 to keep upt 2.5/2.0/enjjpt consistency
    iff_maneuver_grades_df = pd.read_csv(IFF_MANEUVER_GRADE_FILENAME)
    iff_maneuver_grades_df = iff_maneuver_grades_df.loc[pd.to_datetime(iff_maneuver_grades_df['START_DATE_TIME_DT']) >= pd.Timestamp(2019, 6, 18)]
    upt_2_5_students_df = pd.read_csv('data/upt_2_5_students.csv').fillna("")
    upt_2_0_students_df = pd.read_csv('data/upt_2_0_students.csv').fillna("")

    iff_maneuver_grades_df = remove_fake_students(iff_maneuver_grades_df)
    iff_maneuver_grades_df = remove_no_grade_items(iff_maneuver_grades_df)
    iff_maneuver_grades_df = remap_iff_sorties(iff_maneuver_grades_df)
    iff_maneuver_grades_df = remap_iff_items(iff_maneuver_grades_df)

    upt_2_5_students_df = get_ids_for(iff_maneuver_grades_df, upt_2_5_students_df)
    upt_2_5_grades_df = get_maneuver_grades_for(iff_maneuver_grades_df, upt_2_5_students_df)
    average_upt_2_5_grades_df = upt_2_5_grades_df.groupby(['SYL_EVNT_NAME_NM', 'SYL_EVNT_ITM_NAME_NM']).mean()['ITEM_GRADE_QY']
    count_upt_2_5_grades_df = upt_2_5_grades_df.groupby(['SYL_EVNT_NAME_NM', 'SYL_EVNT_ITM_NAME_NM']).count()['ITEM_GRADE_QY']
    average_upt_2_5_grades_df.to_csv('upt_2_5_average.csv')
    count_upt_2_5_grades_df.to_csv('upt_2_5_count.csv')

    upt_2_0_students_df = get_ids_for(iff_maneuver_grades_df, upt_2_0_students_df)
    upt_2_0_grades_df = get_maneuver_grades_for(iff_maneuver_grades_df, upt_2_0_students_df)
    average_upt_2_0_grades_df = upt_2_0_grades_df.groupby(['SYL_EVNT_NAME_NM', 'SYL_EVNT_ITM_NAME_NM']).mean()['ITEM_GRADE_QY']
    count_upt_2_0_grades_df = upt_2_0_grades_df.groupby(['SYL_EVNT_NAME_NM', 'SYL_EVNT_ITM_NAME_NM']).count()['ITEM_GRADE_QY']
    average_upt_2_0_grades_df.to_csv('upt_2_0_average.csv')
    count_upt_2_0_grades_df.to_csv('upt_2_0_count.csv')

    non_enjjpt_students_df = pd.concat([upt_2_5_students_df, upt_2_0_students_df])
    enjjpt_grades_df = get_maneuver_grades_excluding(iff_maneuver_grades_df, non_enjjpt_students_df)
    average_enjjpt_grades_df = enjjpt_grades_df.groupby(['SYL_EVNT_NAME_NM', 'SYL_EVNT_ITM_NAME_NM']).mean()['ITEM_GRADE_QY']
    count_enjjpt_grades_df = enjjpt_grades_df.groupby(['SYL_EVNT_NAME_NM', 'SYL_EVNT_ITM_NAME_NM']).count()['ITEM_GRADE_QY']
    average_enjjpt_grades_df.to_csv('enjjpt_average.csv')
    count_enjjpt_grades_df.to_csv('enjjpt_count.csv')

    print(enjjpt_grades_df)
    print(average_upt_2_5_grades_df)
    print(average_upt_2_0_grades_df)

if __name__ == "__main__":
    run_track_selector()
    #run_upt_iff_analysis()