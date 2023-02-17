import pandas as pd
import configparser as cp
import sqlalchemy as db
import matplotlib.pyplot as plt
from utils import start_connection

# Loading mySQL DB configuration settings from config
config = cp.ConfigParser(interpolation=None)
config.read("C:/IntentDetection/intent-detection-fournet/config.ini")
channel = config.get("Misc", "channel")

def pie_chart(df):

    df = df.sort_values('transcription_id')
    df_pie = df.groupby(["predicted_intent"])["predicted_label"].count().reset_index(name="count")

    labels = df_pie['predicted_intent']
    sizes = df_pie['count']

    #colors
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)#draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)# Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    #plt.show()

def monthly_bar_chart(merged_df):

    # Variables for monthly intents bar chart
    df_month = merged_df.groupby(["year","month","predicted_intent"])["predicted_label"].count().reset_index(name="count")
    pivot_month = pd.pivot_table(data=df_month, index=['month'], columns=['predicted_intent'], values='count')

    ax = pivot_month.plot.bar(stacked=True, color =['lightseagreen', 'tomato'], figsize=(8,6))
    ax.set_title('Monthly Number of Intents', fontsize=15)
    plt.xticks(rotation=0)
    #plt.show()

def weekly_bar_chart(merged_df):

    # Variables for weekly intents bar chart
    df_week = merged_df.groupby(["year","month", "week", "predicted_intent"])["predicted_label"].count().reset_index(name="count")
    pivot_week = pd.pivot_table(data=df_week, index=['week'], columns=['predicted_intent'], values='count')

    ax = pivot_week.plot.bar(stacked=True, color =['lightseagreen', 'tomato'], figsize=(8,6))
    ax.set_title('Weekly Number of Intents', fontsize=15)
    plt.xticks(rotation=0)
    #plt.show()


try:
    engine, connection, metadata = start_connection()

    df = pd.read_sql("SELECT * FROM intent where channel='{}'".format(channel), connection)
    meta_df = pd.read_sql("SELECT * FROM metadata", connection)

    if(df.empty == False):
        # Preprocessing
        meta_df["file_name"] = meta_df["RefID"].str.replace(".am","")
        # Merging meta table with intent table
        merged_df = pd.merge(meta_df, df, on='file_name')

        # Extracting year, month and week from datetime column
        merged_df['year'] = pd.DatetimeIndex(merged_df['StartTime']).year
        merged_df['month'] = pd.DatetimeIndex(merged_df['StartTime']).month
        merged_df['week'] = pd.DatetimeIndex(merged_df['StartTime']).week

        pie_chart(df), monthly_bar_chart(merged_df), weekly_bar_chart(merged_df)
        plt.show()

    else:
        print("Retrieved empty result from database table")

except: 
    print("ERROR - Unable to retrieve information from database successfully")
