import numpy as np
import os
import sys
import keras
from keras.models import load_model
# python test.py /Users/alxperlc/Desktop/ML_hw6/ pred.csv

base_dir = sys.argv[1]
output_path = sys.argv[2]
model_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(model_dir, 'mf_model.h5')
test_path = os.path.join(base_dir, 'test.csv')
users_path = os.path.join(base_dir, 'users.csv')
movies_path = os.path.join(base_dir, 'movies.csv')



def read_data(path,data_type):
	print ('Reading data from ',path)
	with open(path,'r', encoding= 'ISO-8859-1') as f:
		f.readline()
		UserID = []
		Gender = []
		Age = []
		Occupation = []
		Zip_code = []
		MovieID = []
		Rating = []
		Title = []
		Genres = []
		Genres_list = []
		if data_type == 'train':
			for line in f:
				feat = line.split(',')
				UserID.append(int(feat[1]))
				MovieID.append(int(feat[2]))
				Rating.append(int(feat[3]))
			return UserID, MovieID, Rating

		elif data_type == 'test':
			for line in f:
				feat = line.split(',')
				UserID.append(int(feat[1]))
				MovieID.append(int(feat[2]))
			return UserID, MovieID

		elif data_type == 'movies':	
			for line in f:
				start = line.find('::')
				end = line.rfind('::')
				MovieID.append(int(line[:start]))
				Title.append(line[start+2:end])
				Genre = line[end+2:-1].split('|')
				for g in Genre:
					if g not in Genres_list:
						Genres_list.append(g)
				Genres.append(Genre)
			return MovieID, Title, Genres, Genres_list

		elif data_type == 'users':
			for line in f:
				feat = line.split('::')
				UserID.append(int(feat[0]))
				if (feat[1]== 'F'):
					Gender.append(1)
				else:
					Gender.append(0)
				Age.append(int(feat[2]))
				Occupation.append(int(feat[3]))
				Zip_code.append(int(feat[4][:5])) # skip dash
			return UserID, Gender, Age, Occupation, Zip_code


def main():
	te_UserID, te_MovieID = read_data(test_path, 'test')
	us_UserID, us_Gender, us_Age, us_Occupation, us_Zipcode = read_data(users_path, 'users')
	mo_MovieID, mo_Title, mo_Genres, mo_Genres_list = read_data(movies_path, 'movies')
	te_UserID = np.asarray(te_UserID)
	te_MovieID = np.asarray(te_MovieID)

	model = load_model(model_path)
	pred = model.predict([te_UserID, te_MovieID])

	with open('pred.csv','w') as output:
		print ('\"TestDataID\",\"Rating\"',file=output)
		for index,values in enumerate(pred):
			print ('%d,%f'%(index+1,values),file=output)


if __name__=='__main__':
	main()