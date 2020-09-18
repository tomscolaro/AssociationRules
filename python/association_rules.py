import pandas as pd
import numpy as np

class association_rules:
    
    def __init__(self, df, threshold_type = 'support', threshold_level=.3):
        self.df = df
        
        if threshold_type == 'support': 
            self.support = self.support(self.df, threshold_level)     
        else:   
            self.support = self.support(self.df)
        
        if threshold_type == 'confidence': 
             self.confidence  = self.confidence(self.support, threshold_level)    
        else:
            self.confidence  = self.confidence(self.support)
            
        if threshold_type == 'lift':
            self.lift = self.lift(self.confidence, threshold_level)    
        else:
            self.lift = self.lift(self.confidence)
    
        self.rules = self.lift[['antecedent','consequent','antecedent_support',  'consequent_support', 'confidence', 'lift']]
        
        
    def get_rules(self):
        return self.rules
    
    def get_support(self):
        return self.support
    
    def get_confidence(self):
        return self.confidence
        
    def get_lift(self):
        return self.lift

    def support(self, df, min_threshold):
        X = df.values
        num_products = len(df.columns)

        def freq(x, rows):
            out = (np.sum(x, axis=0) / rows)
            return np.array(out).reshape(-1)

        def get_combos(prev_combination):
    
            prev_combo_count = np.unique(prev_combination.flatten())
    
            for prev_combination in prev_combination:
                max_combination = prev_combination[-1]
        
                mask = prev_combo_count > max_combination
                valids = prev_combo_count[mask]
        
                _tuple = tuple(prev_combination)
                for item in valids:
                    yield from _tuple
                    yield item


        support = freq(X, X.shape[0])
        array_col_index = np.arange(X.shape[1])
        support_dict = {1: support[support >= min_threshold]}
        itemset_dict = {1: array_col_index[support >= min_threshold].reshape(-1, 1)}
        max_itemset = 1
        rows_count = float(X.shape[0])


        while max_itemset <= num_products:  
    
            next_max_itemset = max_itemset + 1
    
            combo = get_combos(itemset_dict[max_itemset])
            combo = np.fromiter(combo, dtype=int)
            combo = combo.reshape(-1, next_max_itemset)
    
            bools = np.all(X[:, combo], axis=2)
            support = freq(np.array(bools), rows_count)
            mask = (support >= min_threshold).reshape(-1)
    
            if any(mask):
                itemset_dict[next_max_itemset] = np.array(combo[mask])
                support_dict[next_max_itemset] = np.array(support[mask])
                max_itemset = next_max_itemset
            else:
                break

            res = pd.Series()
            for i in range(max(itemset_dict.keys()),0,-1):
                support = support_dict[i]
                row = list(df.columns[itemset_dict[i]].values)
                res = res.append(pd.Series(row,support))
    
            res = pd.DataFrame(res, columns=['products'])    
            res['products'] = [', '.join(map(str, l)) for l in res['products']]
            res['support'] = res.index
            res = res[['support', 'products']].reset_index(drop=True)

        return res.reset_index(drop=True)
    
    def confidence(self, df, min_threshold=0):
        confidence_df = pd.DataFrame([])

        for i in df['products']:
            
            consq_df = df[df['products'].str.contains(i)]
            antec_support = df[df['products'] == i]['support'].values[0]            
            consq_df['antecedent'] = i
            consq_df['consequent'] = consq_df['products'].str.replace(i, '').str.lstrip(', ').str.rstrip(', ').str.replace(', ,', ', ')
            confidence_df = confidence_df.append(consq_df, ignore_index=True)

        confidence_df.columns= ['confidence_numerator', 'xUy', 'antecedent', 'consequent']
        res = confidence_df.merge(df, left_on ='antecedent', right_on ='products' )
        res = res[[ 'confidence_numerator', 'antecedent', 'support', 'consequent']]
        res.columns = ['confidence_numerator', 'antecedent','antecedent_support', 'consequent']
        res = res.merge(df, left_on ='consequent', right_on ='products' )
        res = res[[  'antecedent','consequent', 'antecedent_support',  'support', 'confidence_numerator']]
        res.columns = ['antecedent','consequent','antecedent_support',  'consequent_support', 'confidence_numerator']
        
    
        res = res.loc[res['consequent'] != '',]
        res['confidence'] = res['confidence_numerator']/res['antecedent_support']

        res = res[res['confidence']>= min_threshold]

        return res.reset_index(drop=True)


    def lift(self, df, min_threshold = 0):
        
        lift_df = df.copy()
        lift_df['lift'] = lift_df['confidence_numerator']/(lift_df['antecedent_support']*lift_df['consequent_support']) 
        return lift_df.reset_index(drop=True) 
