import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import pdb

'''
def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    #return f_list[:-1], int(f_list[-1][-1])
    return f_list[:-1]


def read_csv(data_path, info_path, shuffle=False):
    #D = pd.read_csv(data_path, header=None)
    D= pd.read_csv(data_path,encoding = "ISO-8859-1",delimiter=',')
    D = D.drop('Time', axis=1)
    y_df_series = D['label'].values.reshape(-1,1)
    y_df = pd.DataFrame(y_df_series,columns = ['class']).astype(int)
   
    D['Gender'][D['Gender'] == -1]= 0
    D = D.drop('label', axis=1)
    D = D.drop('MechVent',axis = 1)
    D = D.drop('TroponinI',axis = 1)
    D = D.drop('ICUType',axis = 1)
    D = D.drop('RecordID',axis = 1)
    # can also drop Cholesterol

    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    #f_list, label_pos = read_info(info_path)
    f_list = read_info(info_path)
    X_df = D
    f_df = pd.DataFrame(f_list)
    #D.columns = f_df.iloc[:, 0]
    #y_df = D.iloc[:, [label_pos]]
    #X_df = D.drop(D.columns[label_pos], axis=1)
    #f_df = f_df.drop(f_df.index[label_pos])
    #return X_df, y_df, f_df, label_pos
    return X_df, y_df, f_df

def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    #return f_list[:-1], int(f_list[-1][-1])
    return f_list[:-1]
'''

def read_info(info_path):
    X_df= pd.read_csv(info_path,encoding = "ISO-8859-1",delimiter=' ')
    return X_df

def read_csv(data_path,shuffle=False):
    #D = pd.read_csv(data_path, header=None)
    X_df= pd.read_csv(data_path,encoding = "ISO-8859-1",delimiter=',')
    
    return X_df

def hospital_read_csv(data_path,shuffle=False):
    #D = pd.read_csv(data_path, header=None)
    X_df= pd.read_csv(data_path,encoding = "ISO-8859-1",delimiter=',')
    X_df = X_df.drop('acetohexamide_No',axis = 1)
    X_df = X_df.drop('acetohexamide_infrequent_sklearn',axis = 1)
    X_df = X_df.drop('troglitazone_No',axis = 1)
    X_df = X_df.drop('troglitazone_infrequent_sklearn',axis = 1)
    X_df = X_df.drop('examide_No',axis = 1)
    X_df = X_df.drop('citoglipton_No',axis = 1)
    X_df = X_df.drop('glimepiride-pioglitazone_No',axis = 1)
    X_df = X_df.drop('glimepiride-pioglitazone_infrequent_sklearn',axis = 1)
    X_df = X_df.drop('metformin-pioglitazone_No',axis = 1)
    X_df = X_df.drop('metformin-pioglitazone_infrequent_sklearn',axis = 1)
    return X_df

def college_scorecard_read_csv(data_path,shuffle=False):
    #D = pd.read_csv(data_path, header=None)
    X_df= pd.read_csv(data_path,encoding = "ISO-8859-1",delimiter=',')
    X_df = X_df.drop('sch_deg',axis = 1)
    X_df = X_df.drop('HBCU',axis = 1)
    X_df = X_df.drop('DISTANCEONLY',axis = 1)
    return X_df


def ASSISTments_read_csv(data_path,shuffle=False):
    #D = pd.read_csv(data_path, header=None)
    X_df= pd.read_csv(data_path,encoding = "ISO-8859-1",delimiter=',')
    X_df = X_df.drop('problem_type_choose_n',axis = 1)
    X_df = X_df.drop('problem_type_rank',axis = 1)
    X_df = X_df.drop('problem_type_open_response',axis = 1)
    
    X_df = X_df.drop('type_RandomIterateSection',axis = 1)
    X_df = X_df.drop('type_PlacementsSection',axis = 1)
    X_df = X_df.drop('type_ChooseConditionSection',axis = 1)
    
    return X_df




def continuous_bin(X_df):
    
    discrete_data = X_df[X_df.keys()[self.f_df['type']=='discrete']]
    continuous_data = X_df[X_df.keys()[self.f_df['type']=='continuous']]
    
    
    return X_df

'''
def read_data(dataset,y_train_dataset,DATA_DIR):
    train_data_path = os.path.join(DATA_DIR, dataset + '.csv')
    y_data_path = os.path.join(DATA_DIR, y_train_dataset + '.csv')

    X_df = read_csv(train_data_path, shuffle=True)
    y_df = read_csv(y_data_path, shuffle=True)
    pdb.set_trace()Mi
    #X_train = X_df.to_numpy()
    #y_train = y_df.to_numpy()
    return X_train,y_train
'''


'''
def read_csv(data_path, info_path, shuffle=False):
    #D = pd.read_csv(data_path, header=None)
    D = pd.read_csv(data_path,encoding = "ISO-8859-1",delimiter=',')
    #pdb.set_trace()

    D = D.drop('ROW_ID',axis = 1)
    D = D.drop('HADM_ID',axis = 1)
    D = D.drop('SUBJECT_ID',axis = 1)
    D = D.drop('ICUSTAY_ID',axis = 1)
    D = D.drop('DIAGNOSIS',axis = 1)
    D = D.drop('Capillary refill rate',axis = 1)
    D = D.drop('Fraction inspired oxygen',axis = 1) 
    D = D.drop('ETHNICITY',axis = 1) 
    
    # ETHNICITY   
    # GENDER
    # D['Fraction inspired oxygen'].isna().sum() 5262
    # D['MORTALITY_INUNIT'] is binary
    # D['MORTALITY'] is binary
    
    #pdb.set_trace()
    y_df_series = D['MORTALITY_INHOSPITAL'].values.reshape(-1,1)
    D = D.drop('MORTALITY_INHOSPITAL',axis = 1) 
    y_df = pd.DataFrame(y_df_series,columns = ['class']).astype(int)
   
    D['GENDER'][D['GENDER'] == 'M']= 0
    D['GENDER'][D['GENDER'] == 'F']= 1
   
    D = D.fillna(0)
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
        
    f_list = read_info(info_path)
    #pdb.set_trace()
    X_df = D
   
    D = D.rename(columns={'Diastolic blood pressure': 'Diastolic_blood_pressure', 'Glascow coma scale total': 'Glascow_coma_scale_total','Heart Rate': 'Heart_Rate','Mean blood pressure': 'Mean_blood_pressure','Oxygen saturation': 'Oxygen_saturation','Respiratory rate': 'Respiratory_rate','Systolic blood pressure': 'Systolic_blood_pressure','MORTALITY INUNIT': 'MORTALITY_INUNIT'})
    f_df = pd.DataFrame(f_list)
    #f_df = pd.DataFrame(f_list)

    return X_df, y_df, f_df

def read_data(dataset,DATA_DIR):
    train_data_path = os.path.join(DATA_DIR, dataset + '.csv')
    #info_path = os.path.join(DATA_DIR,  'set_a_ave.info')
    X_df, y_df = read_csv(train_data_path, shuffle=True)
    #X_train = X_df.to_numpy()
    #y_train = y_df.to_numpy()
    return X_df,y_df

'''


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def binning_column(self, dataset, col, num_bins):
        
        traindata = dataset[col] ## Use training data only to determine the bin boundaries
        
        if num_bins == np.inf:
            return np.argsort(traindata)
        else:
            targetdata = dataset[col]
            bins_boundary = np.percentile(traindata, np.arange(0, 100, step=100/num_bins))
            bins_boundary[-1] = max(targetdata)    
            column_value_index = np.digitize(targetdata, bins=bins_boundary[1:], right=False)
            return bins_boundary, column_value_index

        '''
        elif len(np.unique(traindata)) < num_bins:
            pdb.set_trace()
            bins = traindata.unique()
            targetdata = dataset[col]
            
            #return np.digitize(targetdata, bins=bins[1:], right=False)
            return bins, np.digitize(targetdata, bins=bins[1:], right=False)
      
            #bins save the boundary value
            #val_index: return the interval index for each value
            #bins: bin[0] | bin[1]  | bin[2] |...
            #val_i:       0         1        2            
        '''



    def Binning(self, dataset, num_bins, binning_reg=True):

        #continuous_name = dataset.keys()[self.f_df['type']=='continuous']
        continuous_name = dataset.columns
        cell_boundary_index  = pd.DataFrame()
        cell_interval_index =  pd.DataFrame()
    
        
        for col in range(len(continuous_name)):
            col_name = continuous_name[col]
            index = dataset.columns.get_loc(col_name)
            cell_boundary, cell_interval = self.binning_column(dataset, col_name, num_bins)
            cell_boundary_index[col_name] = pd.DataFrame(cell_boundary)
            cell_interval_index[col_name] = pd.DataFrame(cell_interval) 

            
        #pdb.set_trace()
        #binned_dataset = torch.from_numpy(np.stack(binned_dataset, axis=-1)).to(device).type(torch.int64)

        '''
        if binning_reg: ## Do standardization to bin indices
            binned_dataset = binned_dataset.type(torch.float32)
            binned_dataset["stats"] = {'mean': binned_dataset['X_train'].mean(0, keepdim=True)[0], 'std': binned_dataset['X_train'].std(0, keepdim=True)[0]}
            binned_dataset = (binned_dataset - binned_dataset["stats"]["mean"]) / (binned_dataset["stats"]["std"]+1e-10)
        '''
        return cell_boundary_index, cell_interval_index


    def standardization(self, dataset, y=False):
        def prep(data, mean, std):
            return (data - mean) / std

        for col in range(dataset['num_features']):
            if col in dataset["X_num"]:
                dataset['X_train'][:, col] = prep(dataset['X_train'][:, col], mean=dataset['stats']['x_mean'][col], std=dataset['stats']['x_std'][col])
                dataset['X_val'][:, col] = prep(dataset['X_val'][:, col], mean=dataset['stats']['x_mean'][col], std=dataset['stats']['x_std'][col])
                dataset['X_test'][:, col] = prep(dataset['X_test'][:, col], mean=dataset['stats']['x_mean'][col], std=dataset['stats']['x_std'][col])

        if y:
            dataset['y_train'] = prep(dataset['y_train'], mean=dataset['stats']['y_mean'], std=dataset['stats']['y_std'])
            dataset['y_val'] = prep(dataset['y_val'], mean=dataset['stats']['y_mean'], std=dataset['stats']['y_std'])
            dataset['y_test'] = prep(dataset['y_test'], mean=dataset['stats']['y_mean'], std=dataset['stats']['y_std'])

        return dataset


    def split_data(self, X_df):

        discrete_data = X_df[X_df.keys()[self.f_df['type']=='discrete']]
        continuous_data = X_df[X_df.keys()[self.f_df['type']=='continuous']]
        
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(float)

        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)

        discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df)
        self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
        if not discrete_data.empty:

            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            #print ('discrete feature name is:{}'.format(feature_names))
            self.X_fname = list(self.feature_enc.get_feature_names_out(feature_names))

            self.discrete_flen = len(self.X_fname)
            if not self.discrete:
                self.X_fname.extend(continuous_data.columns)
        else:
            self.X_fname = continuous_data.columns
            self.discrete_flen = 0
        self.continuous_flen = continuous_data.shape[1]
        
        

    def transform(self, X_df, y_df, normalized=False, keep_stat=False,num_bins = 5):
        normalized = False
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.

        y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()
        
        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std

        if not discrete_data.empty:
            # dim of discrete_data change here from 7 to 9
            '''
            pdb.set_trace()
            for index in range (len(discrete_data.columns)):
                print (discrete_data.columns[index])
                print (set(discrete_data[discrete_data.columns[index]]))
            ''' 
            discrete_data = self.feature_enc.transform(discrete_data)
            
            if not self.discrete:
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
                continuous_column_index = np.arange(X_df.shape[1] -continuous_data.shape[1] ,X_df.shape[1])
            else:
                X_df = pd.DataFrame(discrete_data.toarray())
        else:
            X_df = continuous_data
            
            continuous_column_index = np.arange(X_df.shape[1])

        cell_boundary_index, cell_interval_index = self.Binning(continuous_data, num_bins, binning_reg=True)

        #print ('transform function')
        #print ('X_df dim is :{}'.format(X_df.shape[1]))
        continuous_flen = continuous_data.shape[1]
        discrete_flen = discrete_data.shape[1]
        return X_df.values, y, cell_boundary_index, cell_interval_index, continuous_column_index,discrete_flen,continuous_flen
