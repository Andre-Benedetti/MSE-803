# Use Case 3 - Product Performance KPIs
# Top-Selling Products: Products contributing the most to sales.
# Top-Selling Products Per Store: Products contributing the most to sales in each store.
# Top-Selling Category: Product Categories contributing the most to sales.
# Top-Selling Category Per Store: Product Category contributing the most to sales in each store.

# Use Case 3.1 - Top-Selling Products: Products contributing the most to sales.
SELECT 
    product_name, 
    SUM(total_price) AS total_sales
FROM Sales
GROUP BY product_name
ORDER BY total_sales DESC LIMIT 3;
# Findings
# Highest sales contribution: Shampoo - $27,041.36
# 2nd Highest sales contribution: Notebook - $24,792.98
# 3rd Highest sales contribution: Orange Juice - $24,686.46

# Use Case 3.2 - Top-Selling Products Per Store: Products contributing the most to sales in each store.
SELECT 
    city,
    product_name,
    SUM(total_price) AS total_sales
FROM Sales
GROUP BY city, product_name
ORDER BY city, total_sales DESC;
# Findings
# Top 2 Sales Contributors for Chicago
# Orange Juice - $10,123.48
# Shampoo - $10,112.73

# Top 2 Sales Contributors for Los Angeles
# Orange Juice - $8,564.29
# Notebook - $7,607.34

# Top 2 Sales Contributors for New York
# Shampoo - $9,879.68
# Notebook - $9,251.19

# Use Case 3.3 - Top-Selling Category: Product Categories contributing the most to sales.
SELECT 
    product_category, 
    SUM(total_price) AS total_sales
FROM Sales
GROUP BY product_category
ORDER BY total_sales DESC;
# Findings 
# Top Category: Personal Care - $27,050.18
# 2nd Top Category: Fruits - $26,197.45
# 3rd Top Category: Beverages - $22,983.32

# Use Case 3.4 - Top-Selling Category: Product Category contributing the most to sales per location.
SELECT city, product_category, total_sales
FROM (
    SELECT 
        city, 
        product_category, 
        SUM(total_price) AS total_sales,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY SUM(total_price) DESC) AS category_rank
    FROM Sales
    GROUP BY city, product_category
) AS ranked
WHERE category_rank = 1
ORDER BY city;
# Findings
# Chicago highest contributing category is Stationary at $9,810.67
# Los Angeles highest contributing category is Fruits at $8,864.57
# New York highest contributing category is Personal Care at $10,175.68

# Use Case 3 Findings

# Overall Company Product Performance KPIs
# Use Case 3.1 shows that Shampoo is the companies highest performing product.
# My recommendation is to capitalize off of that and also order conditioners, body wash, and possibly shower caps. 
# Use Case 3.3 shows that Personal Care  is the highest performing product category.
# My recommendation would be the same as my Use Case 3.1 recommendation.
# It would be useful to pursue Personal Care products outside of bath wear, maybe pursue fragrances or dental products. 

# Product Performances KPIs by Store Locations
# Use Case 3.2 shows that Orange Juice is the top product in both the Chicago and Los Angeles locations.
# My recommendation is to expanding the line of beverages the company has to offer. 
# These 2 stores would be a great place to test the market when selling a new beverage.
# Use Case 3.4 shows that Personal Care is the highest contributing category for the New York location.
# This aligns with Use Case 3.3's findings. If the company pursues the recommendation of expanding their personal care selection, this would be a great location to test the market.