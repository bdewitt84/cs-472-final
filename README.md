# cs-472-final



## Misc notes

survival 	Survival 	0 = No, 1 = Yes
pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
sex 	Sex 	
Age 	Age in years 	
sibsp 	# of siblings / spouses aboard the Titanic 	
parch 	# of parents / children aboard the Titanic 	
ticket 	Ticket number 	
fare 	Passenger fare 	
cabin 	Cabin number 	
embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton

This study suggests min-max may perform better than z-score when using KNN:  
https://ijiis.org/index.php/IJIIS/article/view/73/0

Consider converting 'Fare' column into bins.  
try
```
processor.plot_column_distribution('Fare')
```
to see why

## Installation instructions

Install repository using
```
git clone https://github.com/bdewitt84/cs-472-final.git
```
Navigate to the repository with
```
cd cs-472-final
```

Then install package dependencies using
```
pip install -r requirements.txt
```
Install local packages using
```
pip install -e .
```

## Usage

Train and Validate a K-Nearest Neighbors model on the titanic data in data/ using
```
python src/train_validate.py
```
