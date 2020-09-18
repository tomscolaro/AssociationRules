# Explanation of Association Rules

Often times, data scientists work with a non-technical audiences that have little to no experience in the either computer science, statistics or machine learning. This is okay, because there is a need for variety in roles across a team or company. However, poses interesting challenges in dignifying certain analytics based projects with the value and resources that they deserve. As a result, the burden of socializing the value of these types of projects lies on the data scientists themselves. 

The adoption of these analytics practices is a challenging undertaking for a team or company without a mature approach to tech innovation. Lagging companies often rely on heuristics or anecdotals rather than empirical evidence. One way analytics professionals can make a difference in pushing data driven culture forward is to earn credibility by being clear in explanation of methodology. Too often, data scientists explain esoteric topics with a hand wave and a broad generalization which comes off as disingenuous.

The way to build creditability in these situations is to start with something simple, something to get stakeholders and peers thinking about applicable problems. As an example, in the field of recommenders, association rule mining is a good technique to begin with, because it is simple enough to understand out of the box. There is no, misleading generalizations and can be understood by most with a simple demonstration (something that cannot be said for more advanced techniques). While it may not be as exciting, a successful of campaign or project incorporating these rules in marketing, sales or category management can lead the way in opening people up to additional methodologies. But the trust between the data scientist and the peers needs to be established first.

Now that we can see the value in implementing such rules, what can they be used for? and how is it done? Both of these will be answered by the end of this post, so that way you can begin to implement these into your data science pipeline.

## What is it good for?

Ultimately, association rule mining is a method that creates if-then statements centered around product baskets. As an example, if you were in the grocery store, and you bought peanut butter then you would probably be more likely to also buy jelly. While it seems like an asinine, common sense judgement, these types of relationships exist in all industries and can be leveraged to improve a company operational efficiency. Here are a few of the applications of association rules:

 * Marketing campaigns that your team is riding on so much can be improved in leads generated or click through
 * The sales team trying to find products to upsell can easily be found
 * Supply chain issues you've been having because of inefficient storage 

Generally, these rules are useful in synthesizing alike products and complements. I may do a specific post on the application recommenders, but it out of the scope of this post.    

## The Math

Now to implement the algorithm itself. I am using my version of association rules available [here](https://github.com/tomscolaro/associationrules). There are 3 key calculations that association rules mining accounts for: support, confidence and lift. 

### Support
Support can be thought of as the frequency of occurrence in a transaction set. If apples are present twice in a set of 5 transactions, then the support is .4.  This is done of each combination of products above a threshold, which allows for faster computation. The support of A (a product or a combination in the set N) is:


$
Support(A) = \dfrac{\sigma ({A})}{ \vert T \vert} > SupportThreshold
$

$
A = {N \choose x} 
$


where $\sigma$ denotes the frequency of occurrence in the set. It is also important to note that the number of elements in the combination are $0 < x < { \vert T \vert}$.



### Confidence

Confidence brings the idea of antecedent and consequents into the mix. An antecedent is the product purchased to begin with (the "if" part of the rules), and the consequent would be the product recommended in response to the antecedent (the "then" part of the rule). The confidence is how many times in a transaction set that the rule appears to be true. Statistically, this is likened to the conditional probability $pr(B  \vert  A)$.  The name of the measurement would suggest this is "accuracy", but this isn't necessarily true. As confidence will be higher if the consequent has a high support, regardless of the relationship between the antecedent and consequent.

$
Confidence(A \to\, B) = \dfrac{Support({A \cup B})}{Support(A)} 
$


### Lift

Lift is probably the most important measure of association rules mining. Essentially, it measures how good the rule is. Because lift assumes statistical independence between the antecedent and the consequent in the denominator and uses joint probability in numerator, we can infer whether there is conditional dependence between the antecedent and consequent. 

   * When lift > 1 the antecedent and consequent occur together more frequently, implying that the A and B are dependent and occur at higher frequencies. The higher the better, and we want to use these rules as much as we can.
    
   * When lift < 1 the antecedent and consequent occur together less frequently, implying that the A and B are dependent and occur at lower frequencies. The closer to 0 that the rule is, the worse the rule would perform. However, we can use this to avoid sub optimal decisions. As an example, in a limited shelving space display, avoiding placing poor lift baskets together.

   * When lift ${ \approx }$ 1 the antecedent and consequent occur at about the same frequency. This would suggest that the association of the two rules is just by random chance. 
 

$
Lift(A \to\, B) =\dfrac{Support({A \cup B})}{Support{(A)}* Support{(B)}} 
$


## Implementation

Import the libraries used in the project. association_rules is available [here](https://github.com/tomscolaro/associationrules)


```python
import pandas as pd
import numpy as np
import association_rules as ar
```

Read in the transaction data. Each product is associated with a transaction id where 1 or more products can be purchased at a time.


```python
df = pd.read_csv('../data/sales_data.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SALES_ID</th>
      <th>PRODUCT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>Tortillas</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Eggs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001</td>
      <td>Bacon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001</td>
      <td>Cheddar</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001</td>
      <td>Avocado</td>
    </tr>
  </tbody>
</table>
</div>



This step is just data exploration. The dataset we are using is a mock up, so it is cleaned and ready to go. However, in reality more exploration would be needed to clean and preprocess the data. We can see that beer, milk and cookies all have high occurances in the data set, so we can infer there will be rules including some of them. 


```python
df['PRODUCT'].value_counts().plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbf1902a690>




    
![svg](Association%20Rules_files/Association%20Rules_12_1.svg)
    


For the linked repo, the data must be a boolean dataframe with the products in the columns, where each row signifies a transaction. This can be accomplished by using the pivot pandas method and replacing null values with True/False values. 


```python
df = df.pivot(index='SALES_ID', columns='PRODUCT', values='PRODUCT')
df = pd.DataFrame(np.where(df.isnull(), False, True), columns =df.columns).copy()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>PRODUCT</th>
      <th>Avocado</th>
      <th>Bacon</th>
      <th>Beef</th>
      <th>Beer</th>
      <th>Cheddar</th>
      <th>Chicken</th>
      <th>Chocolate</th>
      <th>Cookies</th>
      <th>Eggs</th>
      <th>Frozen Pizza</th>
      <th>Ice Cream</th>
      <th>Ice Cream Cones</th>
      <th>Milk</th>
      <th>Salsa</th>
      <th>Tortillas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



The association rules object is initiated with the boolean dataframe from the previous step along with threshold type and level. It defaults to support and .3, respectively, but can limited by lift or confidence as well. 


```python
association_rules = ar.association_rules(df, threshold_type='support', threshold_level=.3)
```

We can access the support, confidence or lift rules by using any of the .get_support(), .get_confidence() or .get_lift() methods.



```python
support = association_rules.get_support().sort_values(['support'], ascending=False)
support
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>0.666667</td>
      <td>Milk</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.555556</td>
      <td>Beer</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.444444</td>
      <td>Cookies</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.444444</td>
      <td>Tortillas</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.333333</td>
      <td>Avocado, Beer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.333333</td>
      <td>Bacon, Eggs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.333333</td>
      <td>Beer, Milk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.333333</td>
      <td>Beer, Tortillas</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.333333</td>
      <td>Cookies, Milk</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.333333</td>
      <td>Avocado</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.333333</td>
      <td>Bacon</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.333333</td>
      <td>Beef</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.333333</td>
      <td>Eggs</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we can access the total rules with the .get_rules() method. Viewing the rules from the mock dataset, we can see that Bacon and Eggs is a valuable rule with a lift of 3.0 and a rule to avoid would be Milk and Beer with a lift of 0.9.  


```python
rules = association_rules.get_rules()
rules
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>antecedent_support</th>
      <th>consequent_support</th>
      <th>confidence</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avocado</td>
      <td>Beer</td>
      <td>0.333333</td>
      <td>0.555556</td>
      <td>1.00</td>
      <td>1.800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Milk</td>
      <td>Beer</td>
      <td>0.666667</td>
      <td>0.555556</td>
      <td>0.50</td>
      <td>0.900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tortillas</td>
      <td>Beer</td>
      <td>0.444444</td>
      <td>0.555556</td>
      <td>0.75</td>
      <td>1.350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bacon</td>
      <td>Eggs</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>1.00</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beer</td>
      <td>Avocado</td>
      <td>0.555556</td>
      <td>0.333333</td>
      <td>0.60</td>
      <td>1.800</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Beer</td>
      <td>Milk</td>
      <td>0.555556</td>
      <td>0.666667</td>
      <td>0.60</td>
      <td>0.900</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cookies</td>
      <td>Milk</td>
      <td>0.444444</td>
      <td>0.666667</td>
      <td>0.75</td>
      <td>1.125</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Beer</td>
      <td>Tortillas</td>
      <td>0.555556</td>
      <td>0.444444</td>
      <td>0.60</td>
      <td>1.350</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Eggs</td>
      <td>Bacon</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>1.00</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Milk</td>
      <td>Cookies</td>
      <td>0.666667</td>
      <td>0.444444</td>
      <td>0.50</td>
      <td>1.125</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusion

We have seen why association rule mining is important, some potential applications for a business and an implementation of the analysis in python. Ultimately, the power that can be leveraged out of the analysis is only limited by the creativity of the solution. I urge you to try it for yourself and be inventive in the ways that association rule mining can improve your ML and technology pipelines. In future posts, I hope to build upon this and apply the outputs of the analysis to optimize of resources across marketing, sales and ops applications. 

-Tom Scolaro [LinkedIn](https://www.linkedin.com/in/thomasscolaro)

