# Recommendation-Engine-for-Diversifying-Trade

# Problem Statment

Develop a recommendation engine for firms across the world that help them diversify their imports and exports

# DATA SOURCES

WTO - Billateral trade data for the past 17 years
CPEII - Distance and Gravity data

# Approach

Run tSNE to identify clusters within the data in high dimensional space
Run clusting algorithm (DBSCAN) to identify clusters - currently running for 1 product
Build a Neural Net to unmask relationship between the features and recommend.

# Features

GDP
Distance between countries
Trading routes - currently not incoporated
Output capacity of suppliers
No.Trading partners
Products
Year
Gravity between countries - currently not incoporated
