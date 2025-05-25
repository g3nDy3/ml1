import pandas as pd


# Шаг 1. Загрузка и очистка данных
df = pd.read_csv('train.csv')


# Удаление ненужных столбцов
df.drop(['id', 'people_main', 'bdate', 'education_form', 
         'langs', 'city', 'last_seen', 'occupation_type', 'occupation_name',
           'career_start', 'career_end'], axis=1, inplace=True)


def fill_ed(education_status):
    if education_status == 'Undergraduate applicant':
        return 1
    elif education_status == 'Student (Bachelor\'s)':
        return 2
    elif education_status == 'Alumnus (Bachelor\'s)':
        return 3
    elif education_status == 'Student (Master\'s)':
        return 4
    elif education_status == 'Alumnus (Master\'s)':
        return 5
    elif education_status == 'Candidate of Sciences':
        return 6
    else:
        return 0

def fill_sex(sex):
    if sex == 2:
        return 1
    return 0

def fill_life(life_main): 
    if life_main != 'False':
        return int(life_main)
    else:
        return 9

def fill_l(langs):
    s = langs.split(';')
    return len(s)

df['sex'] = df['sex'].apply(fill_sex) # замена пола на бинарные значения (1 - женский, 0 - мужской)
df['education_status'] = df['education_status'].apply(fill_ed)
df['life_main'] = df['life_main'].apply(fill_life)




# Удаление строк с оставшимися пропущенными значениями
df.dropna(inplace=True)






from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer


X = df.drop('result', axis = 1)
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)



