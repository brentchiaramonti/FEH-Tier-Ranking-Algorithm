#imports a csv using pandas
import pandas

url='data/heroes_data.csv'
names=['Name', 'HP', 'Attack', 'Speed', 'Def', 'Res', 'Total', 'Tier', 'Availability',
       'Color', 'Weapon', 'Movement', 'Legendary', 'Mythic', 'Duo', 'Unique Weapon',
       'Unique Assist', 'Unique Special', 'Unique Skills']
data = pandas.read_csv(url, names=names)

#preprocessing
from sklearn.preprocessing import StandardScaler
import numpy



def handle_non_numerical_data(data):
    columns = data.columns.values
    
    text_digit_vals = {'5_Legacy': 0, 'Grand_Hero_Battle': 1, '4_Star_Story': 2, '5_star': 3, '4_5': 4, 'Tempest_Trial': 5, '3_4': 6, '2': 6,
'Green': 0, 'Blue': 1, 'Red': 2, 'Gray': 3,
'Dagger': 0, 'Staff': 1, 'Sword': 2, 'Lance': 3, 'Bow': 4, 'Dragon': 5, 'Tome': 6, 'Beast': 7, 'Axe': 8,
'Flier': 0, 'Armored': 1, 'Infantry': 2, 'Horse': 3}
    
    for column in columns:
        def convert_to_int(val):
            return text_digit_vals[val]

        if data[column].dtype != numpy.int64 and data[column].dtype != numpy.float64:
            column_contents = data[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            data[column] = list(map(convert_to_int, data[column]))
    return data

def get_user_input():
    hp = int(input("Enter HP: "))
    attack = int(input("Enter Attack: "))
    speed = int(input("Enter Speed: "))
    defense = int(input("Enter Defense: "))
    res = int(input("Enter Res: "))
    total = hp + attack + speed + defense + res
    
    availability = input("Enter Availability (5_Legacy, Grand_Hero_Battle, 4_Star_Story, 5_star, 4_5, Tempest_Trial, 3_4): ")
    color = input("Enter Color (Red, Blue, Green, Grey): ")
    weapon = input("Enter Weapon Type (Sword, Lance, Axe, Tome, Bow, Dagger, Staff, Beast, Dragon): ")
    movement = input("Enter Movement Type (Flier, Armored, Infantry, Horse): ")

    legendary = int(input("Is it a Legendary Unit? (Enter 1 for yes, 0 for no): "))
    mythic = int(input("Is it a Mythic Unit? (Enter 1 for yes, 0 for no): "))
    duo = int(input("Is it a Duo Unit? (Enter 1 for yes, 0 for no): "))
    unique_weapon = int(input("Does the unit have a unique weapon? (Enter 1 for yes, 0 for no): "))
    unique_assist = int(input("Does the unit have a unique assist? (Enter 1 for yes, 0 for no): "))
    unique_special = int(input("Does the unit have a unique special? (Enter 1 for yes, 0 for no): "))
    unique_skill = int(input("Does the unit have a unique skill? (Enter 1 for yes, 0 for no): "))

    return pandas.DataFrame(
    {
    'HP': [hp]
    , 'Attack': [attack]
    , 'Speed': [speed]
    , 'Def': [defense]
    , 'Res': [res]
    , 'Total': [total]
    , 'Availability': [availability]
    , 'Color': [color]
    , 'Weapon': [weapon]
    , 'Movement': [movement]
    , 'Legendary': [legendary]
    , 'Mythic': [mythic]
    , 'Duo': [duo]
    , 'Unique Weapon': [unique_weapon]
    , 'Unique Assist': [unique_assist]
    , 'Unique Special': [unique_special]
    , 'Unique Skills': [unique_skill]
    }
    )


input_x = get_user_input()

data = handle_non_numerical_data(data)

input_x = handle_non_numerical_data(input_x)

array = data.values
X = numpy.concatenate((array[:,1:7], array[:,8:]), axis=1)

Y = array[:,7]
Y = Y.astype('int')


# Evaluate using Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


#computing class weights
from sklearn.utils.class_weight import compute_class_weight

classes=[1,2,3,4,5]

cw = compute_class_weight('balanced', classes, Y)

class_weights = {1:cw[0], 2:cw[1], 3:cw[2], 4:cw[3], 5:cw[4]}

kfold = KFold(n_splits=12, random_state=7, shuffle=True)
model = RandomForestClassifier(n_estimators=100,
                               class_weight=class_weights, max_features='sqrt')

model.fit(X, Y)


print("The unit is tier ", model.predict(input_x.values))
print("This is the probablity of each tier:")
print(model.predict_proba(input_x.values))
