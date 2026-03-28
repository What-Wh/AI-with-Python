import numpy as np
import pandas as pd

df = pd.read_csv('./assets/orders.csv')

df['OrderDate'] = pd.to_datetime(df['OrderDate'])

df['TotalAmount'] = df['Quantity'] * df['Price']

#suma = df['TotalAmount'].sum()
#print(suma)

#avr = df['TotalAmount'].mean()
#print(avr)

#name_quan = df['Customer'] + df['Quantity']
#print(name_quan)

#moreThan500 = df[df['TotalAmount'] > 500]
#print(moreThan500)

#sorted_by_data = df.sort_values(by='OrderDate', ascending=False)
#print(sorted_by_data)

# between_dates = df[
#     (df['OrderDate'] >= '2023-06-05') &
#     (df['OrderDate'] <= '2023-06-10')
# ]
# print(between_dates)

# sorted_by_catagory = df.sort_values(by='Category', ascending=True)
# suma_values_by_catagory = df.groupby('Category').aggregate({'Quantity': 'sum'})
# print(suma_values_by_catagory)
# suma_price_by_category = df.groupby('Category').aggregate({'TotalAmount': 'sum'})
# print(suma_price_by_category)
#print(sorted_by_catagory)

# top_three = df.sort_values(by='TotalAmount', ascending=False).head(3)
# print(top_three)

#print(df)

revenue_per_category = df.groupby('Category')['TotalAmount'].sum()

import matplotlib.pyplot as plt

revenue_per_category.plot(kind='bar', figsize=(10,5), color='skyblue')

plt.title('Розподіл доходів по категоріях')
plt.xlabel('Категорія')
plt.ylabel('Доходи')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.show()