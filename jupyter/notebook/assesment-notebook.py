# Databricks notebook source
# Import Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Initialize Spark
spark = SparkSession.builder \
    .appName("pyspark-assessment") \
    .getOrCreate()

print(f"Spark {spark.version} initialized\n")

# COMMAND ----------

df_raw = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("multiline", "true")
    .option("quote", "\"")
    .option("escape", "\"")
    .option("mode", "PERMISSIVE")
    .option("ignoreLeadingWhiteSpace", "true")
    .option("ignoreTrailingWhiteSpace", "true")
    .load("/nyc-jobs.csv")
)

display(df_raw)


# COMMAND ----------

from pyspark.sql.functions import col, to_timestamp, to_date

df_converted = (
    df_raw
    # Posting Date
    .withColumn(
        "posting_timestamp",
        to_timestamp(col("Posting Date"), "yyyy-MM-dd'T'HH:mm:ss.SSS")
    )
    .withColumn(
        "posting_date",
        to_date(col("posting_timestamp"))
    )

    # Posting Updated
    .withColumn(
        "posting_updated_timestamp",
        to_timestamp(col("Posting Updated"), "yyyy-MM-dd'T'HH:mm:ss.SSS")
    )
    .withColumn(
        "posting_updated_date",
        to_date(col("posting_updated_timestamp"))
    )

    # Process Date
    .withColumn(
        "process_timestamp",
        to_timestamp(col("Process Date"), "yyyy-MM-dd'T'HH:mm:ss.SSS")
    )
    .withColumn(
        "process_date",
        to_date(col("process_timestamp"))
    )
)

df_converted.select(
    "Posting Date",
    "posting_timestamp",
    "posting_date",
    "Posting Updated",
    "posting_updated_timestamp",
    "posting_updated_date",
    "Process Date",
    "process_timestamp",
    "process_date"
).display(5, truncate=False)


# COMMAND ----------

# DBTITLE 1,Data Profiling
from pyspark.sql.functions import col
from pyspark.sql.types import (
    IntegerType, LongType, DoubleType, FloatType, StringType
)

def profile_data(df):
    """Generate comprehensive data profile"""

    # Data types classification
    numerical_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, (IntegerType, LongType, DoubleType, FloatType))
    ]
    string_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, StringType)
    ]

    print(f"\nNumerical columns: {len(numerical_cols)}")
    print(f"String columns: {len(string_cols)}")

    # Null analysis
    print(f"\n{'Column':<35} {'Type':<15} {'Nulls':<10} {'Null %':<8}")
    print("-"*70)

    total = df.count()

    for field in df.schema.fields:
        col_name = field.name
        dtype = field.dataType.simpleString()

        if isinstance(field.dataType, StringType):
            nulls = df.filter(
                col(col_name).isNull() | (col(col_name) == '')
            ).count()
        else:
            nulls = df.filter(col(col_name).isNull()).count()

        null_pct = (nulls / total) * 100
        print(f"{col_name:<35} {dtype:<15} {nulls:<10} {null_pct:>6.1f}%")

    # Categorical cardinality
    print(f"\n{'Categorical Column':<35} {'Unique Values'}")
    print("-"*50)

    for col_name in ['Job Category', 'Agency', 'Posting Type', 'Level', 'Salary Frequency']:
        if col_name in df.columns:
            unique = df.select(col_name).distinct().count()
            print(f"{col_name:<35} {unique:>10,}")

    return {
        'numerical_cols': numerical_cols,
        'string_cols': string_cols,
        'total_rows': total
    }
profile_data(df_converted)

# COMMAND ----------

# DBTITLE 1,Data Cleaning
from pyspark.sql.functions import (
    col, regexp_replace, trim, initcap,
    year, month
)

def clean_data(df):
    """
    Final cleaning step after date conversion
    Assumes posting_date, posting_updated_date, process_date already exist
    """

    # Salary cleaning
    df = (
        df
        .withColumn(
            "salary_from",
            regexp_replace(col("Salary Range From"), r"[^\d.]", "").cast("double")
        )
        .withColumn(
            "salary_to",
            regexp_replace(col("Salary Range To"), r"[^\d.]", "").cast("double")
        )
        .withColumn(
            "salary_avg",
            (col("salary_from") + col("salary_to")) / 2
        )
    )

    # Positions
    df = df.withColumn(
        "num_positions",
        regexp_replace(col("# Of Positions"), r"\D", "").cast("int")
    )

    #  Date dimensions (SAFE  no parsing)
    df = (
        df
        .withColumn("posting_year", year(col("posting_date")))
        .withColumn("posting_month", month(col("posting_date")))
    )

    # Text standardization
    for c in ["Agency", "Job Category", "Business Title"]:
        if c in df.columns:
            df = df.withColumn(c, trim(initcap(col(c))))

    #  Deduplication
    df = df.dropDuplicates(["Job ID"])

    #  Null handling (only where safe)
    df = df.fillna({
        "Agency": "Unknown",
        "Job Category": "Unknown",
        "salary_avg": 0,
        "num_positions": 1
    })

    return df
df_clean = clean_data(df_converted)
display(df_clean)

# COMMAND ----------

# DBTITLE 1,FEATURE ENGINEERING

def engineer_features(df):
    """Apply feature engineering techniques"""
    
    # 1. Salary binning
    df = df.withColumn('salary_category',
        when(col('salary_avg') < 40000, 'Entry')
        .when(col('salary_avg') < 70000, 'Mid')
        .when(col('salary_avg') < 100000, 'Senior')
        .when(col('salary_avg') >= 100000, 'Executive')
        .otherwise('Unknown'))
    
    # 2. Posting age
    df = df.withColumn('posting_age_days', 
                       datediff(current_date(), col('posting_date'))) \
           .withColumn('posting_status',
                       when(col('posting_age_days') <= 30, 'Recent')
                       .when(col('posting_age_days') <= 90, 'Active')
                       .otherwise('Old'))
    
    # 3. Text features
    df = df.withColumn('description_length', length(col('Job Description'))) \
           .withColumn('has_preferred_skills', 
                       when(col('Preferred Skills').isNotNull(), 1).otherwise(0)) \
           .withColumn('is_fulltime',
                       when(col('Full-Time/Part-Time indicator').like('%F%'), 1).otherwise(0))
    
    # 4. Education extraction
    df = df.withColumn('requires_phd',
                       when(lower(col('Minimum Qual Requirements')).like('%phd%'), 1).otherwise(0)) \
           .withColumn('requires_masters',
                       when(lower(col('Minimum Qual Requirements')).like('%master%'), 1).otherwise(0)) \
           .withColumn('requires_bachelors',
                       when(lower(col('Minimum Qual Requirements')).like('%bachelor%'), 1).otherwise(0)) \
           .withColumn('education_level',
                       when(col('requires_phd') == 1, 'PhD')
                       .when(col('requires_masters') == 1, 'Masters')
                       .when(col('requires_bachelors') == 1, 'Bachelors')
                       .otherwise('High School'))
    
    # 5. Aggregated features
    agency_stats = df.groupBy('Agency').agg(avg('salary_avg').alias('agency_avg_salary'))
    df = df.join(agency_stats, 'Agency', 'left') \
           .withColumn('salary_vs_agency_avg', col('salary_avg') - col('agency_avg_salary'))
    
    print("Features engineered")
    return df

df_featured = engineer_features(df_clean)

# COMMAND ----------

display(df_featured)

# COMMAND ----------

# DBTITLE 1,FEATURE SELECTION
def select_features(df):
    """Remove high-null and redundant columns"""
    
    cols_to_drop = [
        'Recruitment Contact', 'Residency Requirement', 'Additional Information',
        'To Apply', 'Hours/Shift', 'Division/Work Unit', 'Work Location 1',
        'Title Code No', 'Posting Updated', 'Process Date',
        'Salary Range From', 'Salary Range To', '# Of Positions',
        'Posting Date', 'Post Until'
    ]
    
    df = df.drop(*[c for c in cols_to_drop if c in df.columns])
    print(f"Selected {len(df.columns)} features")
    return df

df_final = select_features(df_featured)

# COMMAND ----------

display(df_final)

# COMMAND ----------

# MAGIC %md
# MAGIC KPI

# COMMAND ----------

# DBTITLE 1,Top 10 job categories
# KPI 1: Top 10 job categories
def kpi1_job_postings_by_category(df):
    result = df.groupBy('Job Category') \
               .count() \
               .orderBy(desc('count')) \
               .limit(10) \
               .toPandas()
    
    plt.figure(figsize=(10, 6))
    plt.barh(result['Job Category'], result['count'], color='steelblue')
    plt.xlabel('Number of Postings')
    plt.title('Top 10 Job Categories')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('kpi1_categories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return result
kpi1_job_postings_by_category(df_final)

# COMMAND ----------

# KPI 2: Salary distribution by category
def kpi2_salary_distribution(df):
    result = df.filter(col('salary_avg') > 0) \
               .groupBy('Job Category') \
               .agg(
                   avg('salary_avg').alias('avg_salary'),
                   min('salary_avg').alias('min_salary'),
                   max('salary_avg').alias('max_salary')
               ) \
               .orderBy(desc('avg_salary')) \
               .limit(10) \
               .toPandas()
    
    plt.figure(figsize=(10, 6))
    plt.barh(result['Job Category'], result['avg_salary'], color='coral')
    plt.xlabel('Average Salary ($)')
    plt.title('Salary Distribution by Category')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('kpi2_salary_dist.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return result
kpi2_salary_distribution(df_final)

# COMMAND ----------

# KPI 3: Education vs Salary correlation
def kpi3_education_salary_correlation(df):
    result = df.filter(col('salary_avg') > 0) \
               .groupBy('education_level') \
               .agg(avg('salary_avg').alias('avg_salary')) \
               .orderBy(desc('avg_salary')) \
               .toPandas()
    
    plt.figure(figsize=(10, 6))
    order = ['PhD', 'Masters', 'Bachelors', 'High School']
    result_sorted = result.set_index('education_level').reindex(order).reset_index()
    
    plt.bar(
    result_sorted['education_level'],
    result_sorted['avg_salary'])

    plt.ylabel('Average Salary ($)')
    plt.title('Education Level vs Salary')
    plt.tight_layout()
    plt.savefig('kpi3_education_salary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nCorrelation: Higher education - Higher salary ")
    return result

kpi3_education_salary_correlation(df_final)

# COMMAND ----------

# KPI 4: Highest salary per agency
def kpi4_highest_salary_per_agency(df):
    window = Window.partitionBy('Agency').orderBy(desc('salary_avg'))
    
    result = df.filter(col('salary_avg') > 0) \
               .withColumn('rank', row_number().over(window)) \
               .filter(col('rank') == 1) \
               .select('Agency', 'Business Title', 'salary_avg') \
               .orderBy(desc('salary_avg')) \
               .limit(10) \
               .toPandas()
    
    plt.figure(figsize=(10, 6))
    plt.barh(result['Agency'], result['salary_avg'], color='darkgreen')
    plt.xlabel('Highest Salary ($)')
    plt.title('Top Paying Job per Agency')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('kpi4_highest_per_agency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return result
kpi4_highest_salary_per_agency(df_final)

# COMMAND ----------

from pyspark.sql.functions import col, avg, desc, max as spark_max
import matplotlib.pyplot as plt

def kpi5_avg_salary_last_2_years(df):

    max_year = (
        df.select(spark_max(col("posting_year")).alias("max_year"))
          .collect()[0]["max_year"]
    )

    result = (
        df.filter(
            (col("posting_year") >= max_year - 2) &
            (col("salary_avg") > 0)
        )
        .groupBy("Agency")
        .agg(avg("salary_avg").alias("avg_salary"))
        .orderBy(desc("avg_salary"))
        .limit(10)
        .toPandas()
    )

    plt.figure(figsize=(10, 6))
    plt.barh(result["Agency"], result["avg_salary"])
    plt.xlabel("Average Salary ($)")
    plt.title(f"Average Salary per Agency (Last 2 Years in Data: {max_year-2}-{max_year})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return result


kpi5_avg_salary_last_2_years(df_final)

# COMMAND ----------

def kpi6_highest_paid_skills(df):
    skills = ['python', 'java', 'sql', 'aws', 'azure', 'machine learning',
              'data science', 'kubernetes', 'docker', 'react', 'leadership',
              'project management', 'analytics', 'cybersecurity']
    
    skill_data = []

    df_filtered = df.filter(col('salary_avg') > 0)
    
    for skill in skills:
        skill_df = df_filtered.filter(
            lower(col('Preferred Skills')).like(f'%{skill}%') |
            lower(col('Minimum Qual Requirements')).like(f'%{skill}%') |
            lower(col('Job Description')).like(f'%{skill}%')
        )
        
        stats = skill_df.agg(avg('salary_avg').alias('avg_salary')).collect()
        count = skill_df.count()
        
        if count > 0:
            skill_data.append({
                'Skill': skill.title(),
                'avg_salary': stats[0]['avg_salary'],
                'count': count
            })
    
    result = pd.DataFrame(skill_data).sort_values('avg_salary', ascending=False).head(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].barh(result['Skill'], result['avg_salary'])
    axes[0].set_xlabel('Average Salary ($)')
    axes[0].set_title('Highest Paid Skills')
    axes[0].invert_yaxis()
    
    axes[1].barh(result['Skill'], result['count'])
    axes[1].set_xlabel('Job Count')
    axes[1].set_title('Skill Demand')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('kpi6_skills.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return result
kpi6_highest_paid_skills(df_final)

# COMMAND ----------

# Execute KPIs
print("KPI ANALYSIS")

kpi1 = kpi1_job_postings_by_category(df_final)
kpi2 = kpi2_salary_distribution(df_final)
kpi3 = kpi3_education_salary_correlation(df_final)
kpi4 = kpi4_highest_salary_per_agency(df_final)
kpi5 = kpi5_avg_salary_last_2_years(df_final)
kpi6 = kpi6_highest_paid_skills(df_final)

# COMMAND ----------

def save_outputs(df):
    """Save processed data and KPI results"""
    
    # Save processed data
    df.write.mode('overwrite').parquet('/nyc-jobs.csv')
    
    print(f"Saved {df.count():,} processed records")

save_outputs(df_final)

# COMMAND ----------

def run_tests():
    """Execute validation tests"""
    
    tests_passed = 0
    tests_total = 8
    
    # Test 1: Data loaded
    try:
        assert df_raw.count() > 0
        print(" Test 1: Data loaded")
        tests_passed += 1
    except:
        print(" Test 1: Failed")
    
    # Test 2: Salary cleaning
    try:
        assert 'salary_avg' in df_clean.columns
        assert df_clean.select('salary_avg').dtypes[0][1] == 'double'
        print(" Test 2: Salary cleaned")
        tests_passed += 1
    except:
        print(" Test 2: Failed")
    
    # Test 3: Features engineered
    try:
        assert all(c in df_featured.columns for c in ['salary_category', 'education_level'])
        print(" Test 3: Features engineered")
        tests_passed += 1
    except:
        print(" Test 3: Failed")
    
    # Test 4: No nulls in critical columns
    try:
        assert df_final.filter(col('Job Category').isNull()).count() == 0
        print(" Test 4: No nulls in critical columns")
        tests_passed += 1
    except:
        print(" Test 4: Failed")
    
    # Test 5: No duplicates
    try:
        assert df_final.count() == df_final.select('Job ID').distinct().count()
        print(" Test 5: No duplicates")
        tests_passed += 1
    except:
        print(" Test 5: Failed")
    
    # Test 6: Salary ranges valid
    try:
        invalid = df_final.filter(
            (col('salary_to') < col('salary_from')) & (col('salary_from') > 0)
        ).count()
        assert invalid == 0
        print(" Test 6: Salary ranges valid")
        tests_passed += 1
    except:
        print(" Test 6: Failed")
    
    # Test 7: Dates parsed
    try:
        assert 'posting_date' in df_final.columns
        print(" Test 7: Dates parsed")
        tests_passed += 1
    except:
        print(" Test 7: Failed")
    

run_tests()

# SUMMARY
print("PIPELINE SUMMARY")
print(f"Original records: {profile['total_rows']:,}")
print(f"Final records: {df_final.count():,}")
print(f"Features: {len(df_final.columns)}")
print(f"KPIs analyzed: 6")
