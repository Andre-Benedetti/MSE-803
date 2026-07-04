# Use Case 1 - Sales and Revenue KPI's
# Total Revenue: The total amount of money earned from sales.
# Average Basket Size: The average amount spent by customers per transaction.

# Use Case 1.1 - Find the Total Revenue by city and branch.

# Total Revenue by City
SELECT 
    city,
    ROUND(SUM(total_price),0) AS total_revenue_per_location
FROM Sales
GROUP BY city;
# Findings 
# Total Revenue for New York = 40,227
# Total Revenue for Los Angeles = 35,772
# Total Revenue for Chicago = 42,585

# Total Revenue by Branch
SELECT 
    branch,
    ROUND(SUM(total_price),0) AS total_revenue_per_branch
FROM Sales
GROUP BY branch;
# Findings 
# Total Revenue for Branch A = 82,812
# Total Revenue for Branch B = 35,772

# Use Case 1.2 - Find the average basket size of all orders for each City and Branch

# Find the average basket size of all orders in each city location.
 SELECT 
    city,
    ROUND(AVG(quantity),1) AS average_quantity_per_location
FROM Sales
GROUP BY city;
# Findings
# Average basket size per customer in New York = 9.7
# Average basket size per customer in Los Angeles = 10.3
# Average basket size per customer in Chicago = 11.0

# Find the average basket size of all orders in each branch.
 SELECT 
    branch,
    ROUND(AVG(quantity),1) AS average_quantity_per_branch
FROM Sales
GROUP BY branch;
# Findings
# Average basket size per customer in Branch A = 10.4
# Average basket size per customer in Branch B = 10.3

# Use Case 1 Findings
# Chicago is the best performing store by total revenue and also has the highest average items per customer.
# Despite being the only store in Branch B the Los Angeles location has around the same as both stores in Branch A in average items per customer.
# This shows that if the company was thinking of expanding to two stores in Branch B than one can expect a slightly below average performing store.
