import unittest
import pandas as pd
from app.preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_basic(self):
        data = {
            'Item_Identifier': ['FDA15', 'DRC01'],
            'Item_Weight': [9.3, None],
            'Item_Fat_Content': ['Low Fat', 'reg'],
            'Item_Visibility': [0.016, 0.019],
            'Item_Type': ['Dairy', 'Soft Drinks'],
            'Outlet_Identifier': ['OUT049', 'OUT018'],
            'Outlet_Establishment_Year': [1999, 1987],
            'Outlet_Size': [None, 'Medium'],
            'Outlet_Location_Type': ['Tier 1', 'Tier 3'],
            'Outlet_Type': ['Supermarket Type1', 'Supermarket Type2']
        }
        df = pd.DataFrame(data)
        processed = preprocess_data(df, is_train=False)
        self.assertFalse(processed.isnull().any().any())

if __name__ == '__main__':
    unittest.main()
