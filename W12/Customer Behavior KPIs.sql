# Use Case 2 - Customer Behavior KPIs
# Customer Footfall: The number of customers visiting each store.
# Loyalty: Percentage of membership for each store.
# Gender Bias: Percentage of Male and percentage of Female customers per store.
# Big Spenders: Total numbers of transactions over $100 per store.

# Use Case 2.1 - Customer Footfall: The total number of customers visiting each store.
SELECT 
    city,
    COUNT(DISTINCT sale_id) AS total_customers
FROM Sales
GROUP BY city;
# Findings
# Total Customers for Chicago location = 330
# Total Customers for Los Angeles location = 326
# Total Customers for New York location = 344

# Use Case 2.2 - Loyalty: Percentage of membership for each store.
SELECT 
    city,
    ROUND((SUM(CASE WHEN customer_type = 'Member' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 1) AS percentage_members
FROM Sales
GROUP BY city;
# Findings
# Percentage of Customers that are Members for New York = 51.5%
# Percentage of Customers that are Members for Los Angeles = 49.4%
# Percentage of Customers that are Members for Chicago = 53.9%

# Use 2.3 - Gender Bias: Percentage of Male and percentage of Female customers per store.
SELECT 
    city,
    ROUND((SUM(CASE WHEN gender = 'Male' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 1) AS percentage_of_male_customers,
    ROUND((SUM(CASE WHEN gender = 'Female' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 1) AS percentage_of_female_customers
FROM Sales
GROUP BY city;
# Findings
# Male vs Female Customers for New York location = 50% Male and 50% Female
# Male vs Female Customers for Los Angeles location = 52.8% Male and 47.2% Female
# Male vs Female Customers for Chicago location = 55.8% Male and 44.2% Female

# Use Case 2.4 - Big Spenders: Total numbers of transactions over $100 per store.
SELECT 
    city,
    COUNT(*) AS orders_over_100_dollars
FROM Sales
WHERE total_price > 100
GROUP BY city;
# Findings
# Total Big Spender Transactions for Chicago location = 170
# Total Big Spender Transactions for Los Angeles location = 136
# Total Big Spender Transactions for New York location = 160

# Use Case 2 Findings

# New York Location Customer Behavior KPIs Conclusions.
# New York has the highest amount of customers out of all locations, but is behind Chicago in Membership Percentage and Big Spenders.
# If I were to recommend improvements for New York, it would be to increase their Membership push, more Members = more returning customers.

# Chicago Location Customer Behavior KPIs Conclusions.
# Chicago rates the highest in Membership Percentage and Big Spenders, but is second behind New York in total customers.
# If I were to recommend improvements for Chicago, it would be to increase the amount of female centric products.
# The numbers show that only 44.2% of customers are female, so the store can gain more female customers with these products.
# More female products = more total customers.

# Los Angeles Location Customer Behavior KPIs Conclusions.
# Los Angeles is the lowest performing store, so there is tons of room for improvement. 
# Los Angeles biggest area of improvement is with the amount of Big Spenders. Despite only having 4 less total customers than Chicago.
# Despite only having 4 less total customers than Chicago, Los Angeles has 24 less Big Spenders than Chicago.
# To improve this Los Angeles could advertise more expensive items at the front of the store.
# However I think the answer is to boost their Membership count. 
# That way when Members need to buy one of these big ticket items, they would be more likely to choose this store so they can earn more rewards points. 