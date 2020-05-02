import pandas as pd
from utils.utils import get_single_col_by_input_type
from utils.utils import extract_cols_from_data_type
from .base import GenericDataFormatter
from .base import DataTypes, InputTypes
from sklearn.preprocessing import StandardScaler, LabelEncoder

class M5Formatter(GenericDataFormatter):
    """Defines and formats data for the electricity dataset.
    Note that per-entity z-score normalization is used here, and is implemented
    across functions.
    Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
      ('id', DataTypes.CATEGORICAL, InputTypes.ID),
      ('sales', DataTypes.REAL_VALUED, InputTypes.TARGET),
        
      ('d', DataTypes.CATEGORICAL, InputTypes.TIME),
      ('event_name_2', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      ('event_type_2', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      ('event_name_1', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      ('event_type_1', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      ('snap_WI', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('snap_CA', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('snap_TX', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        
      ('price_momentum', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('price_momentum_m', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('price_momentum_y', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),

      ('price_max', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('price_min', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('price_std', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('price_mean', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('price_norm', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('price_nunique', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('sell_price', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        
      ('tm_d', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('tm_w', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('tm_m', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('tm_y', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('tm_wm', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('tm_dw', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('tm_w_end', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        
      ('snap_TX', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('snap_CA', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('snap_WI', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        
      ('item_nunique', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('release', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT), 
      ('item_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT), 
      ('dept_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('cat_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('store_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('state_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self._time_steps = self.get_fixed_params()['total_time_steps']
        self._num_encoder_steps = self.get_fixed_params()['num_encoder_steps']
        
    def get_time_steps(self):
        return self.get_fixed_params()['total_time_steps']
    
    def get_num_encoder_steps(self):
        return self.get_fixed_params()['num_encoder_steps']

    def split_data(self, df, valid_boundary=1913-(72*2 + 28)+1, test_boundary=1913-72+1):
        """Splits data frame into training-validation-test data frames.
        This also calibrates scaling object, and transforms data for each split.
        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data
        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        index = df['d']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
        test = df.loc[index >= test_boundary]

        self.set_scalers(train)
        
        print('Transforming the data')
        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.
            Args:
              df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Format real scalers
        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        print('Real Scalers')
        def create_scalers(x):

            data = x[real_inputs].values
            targets = x[[target_column]].values

            real_scaler = StandardScaler().fit(data)
            target_scaler = StandardScaler().fit(targets)

            return real_scaler, target_scaler
        
        #scalers = df.groupby(id_column).apply(lambda x: pd.Series(create_scalers(x)),
        #                                      meta=[(0, object),
        #                                            (1, object)]).compute().rename(columns={0:'real',
        #                                                                                    1:'target'})
        scalers = m5_data[[id_column] + 
                          real_inputs +
                          [target_column]].groupby(id_column) \
        .apply(lambda x: pd.Series(create_real_scalers(x))) \
        .rename(columns = {0:'real',
                           1: 'target'})

        self._real_scalers = scalers.real.to_dict()
        self._target_scaler = scalers.target.to_dict()
        # Extract identifiers in case required
        self.identifiers = scalers.index.values
        
        # Format categorical scalers
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        
        print('Categorical Scalers')
        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            srs = m5_data[col]
            if srs.dtype.name != 'category':
                # Set all to str so that we don't have mixed integer/string columns
                srs = srs.astype('category')
            categorical_scalers[col] = LabelEncoder().fit(
              srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes
        

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data frame.
        """
        print('Transforming the training data...')
        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        # Extract relevant columns
        column_definitions = self.get_column_definition()
        id_col = get_single_col_by_input_type(InputTypes.ID,
                                                    column_definitions)
        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Transform real inputs per entity
        df_list = []
        print('Real Features Transform')
        for identifier, sliced in df.groupby(id_col):
            print('{} - {}'.format(identifier, len(sliced)))
            # Filter out any trajectories that are too short
            if len(sliced) >= self._time_steps:
                
                sliced_copy = sliced.copy()
                sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
                    sliced_copy[real_inputs].values)
                df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)
        
        output = df[[id_col] + 
                     real_inputs].groupby(id_column) \
        .apply(lambda x: pd.DataFrame(self._real_scalers[x.name].transform(x[real_inputs].values)))
        output = output.droplevel(level=None).reset_index()
        output.columns = [id_col] + real_inputs 

        print('Categorical Features Transform')
        # Format categorical inputs
        for col in categorical_inputs:
            print(col)
            string_df = df[col]#.apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.
        Args:
          predictions: Dataframe of model predictions.
        Returns:
          Data frame of unnormalised predictions.
        """

        if self._target_scaler is None:
            raise ValueError('Scalers have not been set!')

        column_names = predictions.columns

        df_list = []
        for identifier, sliced in predictions.groupby('identifier'):
            sliced_copy = sliced.copy()
            target_scaler = self._target_scaler[identifier]

            for col in column_names:
                if col not in {'forecast_time', 'identifier'}:
                    sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col])
                    df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 72 + 28,
            'num_encoder_steps': 72,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.1,
            'hidden_layer_size': 160,
            'learning_rate': 0.001,
            'minibatch_size': 64,
            'max_gradient_norm': 0.01,
            'num_heads': 4,
            'stack_size': 1
        }

        return model_params

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.
        Use to sub-sample the data for network calibration and a value of -1 uses
        all available samples.
        Returns:
          Tuple of (training samples, validation samples)
        """
        return 450000, 50000