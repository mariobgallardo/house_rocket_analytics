import geopandas
import pandas as pd
import streamlit as st
import numpy as np
import folium
import plotly.express as px

from datetime import datetime
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
st.set_page_config( layout='wide')

@st.cache( allow_output_mutation=True )
def get_data( path ):
    data = pd.read_csv( path )

    return data



@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    geofile = geopandas.read_file( url )

    return geofile



def get_df_i(data):
    df_i = data.copy()

    return df_i



def overview_data(data):
    st.sidebar.title("House Rocket's Project")
    st.sidebar.write(
        ('Data Visualization of 10505 recommended properties to buy, including their selling prices and profits.'))
    st.sidebar.write('Filters for properties based on the two most relevant features:')

    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())
    f_condition = st.sidebar.multiselect('Enter condition', data['condition'].unique())

    st.title("Properties Recommendation")

    if (f_zipcode != []) & (f_condition != []):
        data = data.loc[(data['zipcode'].isin(f_zipcode)) & (data['condition'].isin(f_condition))]
    elif (f_zipcode != []) & (f_condition == []):
        data = data.loc[data['zipcode'].isin(f_zipcode)]
    elif (f_zipcode == []) & (f_condition != []):
        data = data.loc[data['condition'].isin(f_condition)]
    else:
        data = data.copy()

    # calculating profit
    profit = data['profit'].sum()

    st.sidebar.write('There are {} properties to buy based on the conditions above.'.format(data.shape[0]))
    st.sidebar.write(f'The total profit would be: US$ {profit:,.2f}')

    st.subheader('Properties to Buy')
    st.write(
        "Based on two parameters: If condition is regular or good AND if the price is lower than the zipcode's median price ")
    st.dataframe(data[['id', 'zipcode', 'price', 'median price', 'condition']])

    st.subheader('Properties to Sell')
    st.write("""For properties bought, there are two conditions for sale: 
             \n a)properties bought out of spring season shall be sold with 30% over the purchase price 
             \n b)properties bought on spring season shall be sold with 10% over the purchase price""")

    st.dataframe(data[['id', 'zipcode', 'price', 'median price', 'season', 'sell price', 'profit']])

    c1, c2 = st.columns((1, 1))

    # Average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['median price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['profit', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge dataframes
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'MEAN PRICE', 'MEDIAN PRICE', 'PROFIT']

    c1.subheader('Average Values')
    c1.dataframe(df, height=600)

    # Descriptive statistics
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    num_attributes.drop('Unnamed: 0', axis=1, inplace=True)
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()

    df1.columns = ['ATTRIBUTES', 'MAX', 'MIN', 'MEAN', 'MEDIAN', 'STD']

    c2.subheader('Descriptive Statistics')
    c2.dataframe(df1, height=600)

    return None



def portfolio_density(data,geofile):
    st.title('Region Overview')

    c1, c2 = st.columns((1, 1))
    c1.header('Portfolio Density')

    # Base Map - Folium
    density_map = folium.Map(location=[data['lat'].mean(),
                                       data['long'].mean()],
                             default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in data.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R$ {0} on: {1}, Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                          row['price'],
                          row['date'],
                          row['sqft_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Region Price Map
    c2.header('Profit Density')

    df = data[['profit', 'zipcode']].groupby('zipcode').mean().reset_index()

    df.columns = ['ZIP', 'PROFIT']

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()],
                                  default_zoom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PROFIT'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PROFIT')

    with c2:
        folium_static(region_price_map)

    return None



def commercial( data ):
    st.title('Data Visualization')
    st.write('Visualizing Commercial Attributes')

    # ----------- Histogram
    st.header('Profit Distribution')

    # Plot
    fig2 = px.histogram(data, x='profit', nbins=50)
    st.plotly_chart(fig2, use_container_width=True)

    return None



def insights( df_i ):
    st.header('Insights')

    # ZIPCODE
    st.subheader('Zipcode x Profit')
    st.write('Zipcode 98052 has the greatest profit gain')
    df_i1 = df_i[['zipcode', 'profit']].groupby('zipcode').sum().reset_index()
    fig_zipcode = px.bar(df_i1, x='zipcode', y='profit', color='zipcode')
    st.plotly_chart(fig_zipcode, use_container_width=True)

    # CONDITION
    st.subheader('Condition x Profit')
    st.write('Regular condition has the greatest profit gain')
    df_i2 = df_i[['condition', 'profit']].groupby('condition').sum().reset_index()
    fig_cond = px.bar(df_i2, x='condition', y='profit', color='condition')
    st.plotly_chart(fig_cond, use_container_width=True)

    # GRADE
    st.subheader('Grade x Profit')
    st.write('Average design has the greatest profit gain')
    df_i3 = df_i[['grade', 'profit']].groupby('grade').sum().reset_index()
    fig_grade = px.bar(df_i3, x='grade', y='profit', color='grade')
    st.plotly_chart(fig_grade, use_container_width=True)

    # FLOORS
    st.subheader('Floors x Profit')
    st.write('Properties with 1 floor has the greatest profit gain')
    df_i4 = df_i[['floors', 'profit']].groupby('floors').sum().reset_index()
    fig_floors = px.bar(df_i4, x='floors', y='profit', color='floors')
    st.plotly_chart(fig_floors, use_container_width=True)

    # BEDROOMS
    st.subheader('Bedrooms x Profit')
    st.write('Properties with 3 floors has the greatest profit gain')
    df_i5 = df_i[['bedrooms', 'profit']].groupby('bedrooms').sum().reset_index()
    fig_bedrooms = px.bar(df_i5, x='bedrooms', y='profit', color='bedrooms')
    st.plotly_chart(fig_bedrooms, use_container_width=True)

    # BATHROOMS
    st.subheader('Bathrooms x Profit')
    st.write('Properties with 1 bathroom has the greatest profit gain')
    df_i6 = df_i[['bathrooms', 'profit']].groupby('bathrooms').sum().reset_index()
    fig_bathrooms = px.bar(df_i6, x='bathrooms', y='profit', color='bathrooms')
    st.plotly_chart(fig_bathrooms, use_container_width=True)

    # WATERFRONT
    st.subheader('Waterfront x Profit')
    st.write('Properties without waterfront view has the greatest profit gain')
    df_i7 = df_i[['waterfront', 'profit']].groupby('waterfront').sum().reset_index()
    fig_waterfront = px.bar(df_i7, x='waterfront', y='profit', color='waterfront')
    st.plotly_chart(fig_waterfront, use_container_width=True)

    # YR_BUILT
    st.subheader('Year Built x Profit')
    st.write('Properties built on 1977 has the greatest profit gain')
    df_i8 = df_i[['yr_built', 'profit']].groupby('yr_built').sum().reset_index()
    fig_yr_built = px.bar(df_i8, x='yr_built', y='profit', color='yr_built')
    st.plotly_chart(fig_yr_built, use_container_width=True)

    return None



def hyphotesys(df_h):
    st.header('Hypotheses Validation')

    c1, c2 = st.columns((1, 1))

    # H1 - Properties with a good condition and waterfront view are 50 % more expensive
    # on average than properties with good condition but without waterfront view
    c1.subheader("H1 - Good condition's properties with waterfront view are 50% more expensive"
                 ' than without waterfront view')
    c1.write('Actually, they are 188,12% more expensive')

    aux1 = df_h.loc[df_h['condition'] == 'good']
    h1 = aux1[['price', 'waterfront']].groupby('waterfront').mean().reset_index()

    fig_h1 = px.bar(h1, x='waterfront', y='price', color='waterfront')
    c1.plotly_chart(fig_h1, use_container_width=True)

    # H2 - Properties with building year before 1955 are cheaper on average
    c2.subheader('H2 - Properties with building year before 1955 are cheaper on average')
    c2.write("FALSE: The difference is minimal (0,4%), so we can't confirm that")

    aux2 = df_h.copy()
    aux2['construction'] = aux2['yr_built'].apply(lambda x: 'before_1955' if x < 1955 else
    'after_1955')
    h2 = aux2[['price', 'construction']].groupby('construction').mean().reset_index()

    fig_h2 = px.bar(h2, x='construction', y='price', color='construction')
    c2.plotly_chart(fig_h2, use_container_width=True)

    c3, c4 = st.columns((1, 1))
    # H3 - Properties without basement have a square footage of the land space 15% bigger
    # than properties with it
    c3.subheader(
        'H3 - Properties without basement have a square footage of the land space 15% bigger than properties with it')
    c3.write('TRUE: They are 20,62% bigger')

    aux3 = df_h.copy()
    aux3['basement'] = aux3['sqft_basement'].apply(lambda x: 'with basement' if x > 0 else
    'without basement')
    h3 = aux3[['sqft_lot', 'basement']].groupby('basement').mean().reset_index()

    fig_h3 = px.bar(h3, x='basement', y='sqft_lot', color='basement')
    c3.plotly_chart(fig_h3, use_container_width=True)

    # H4 - The propertie's price growth YoY (year over year) it's 10%
    c4.subheader("H4 - The propertie's price growth YoY (year over year) it's 10%")
    c4.write("FALSE: The price growth YoY haven't change (difference of 0,18%) ")

    aux4 = df_h.copy()
    aux4['year'] = pd.DatetimeIndex(aux4['date']).year

    h4 = aux4[['price', 'year']].groupby('year').mean().reset_index()

    fig_h4 = px.bar(h4, x='year', y='price', color='year')
    c4.plotly_chart(fig_h4, use_container_width=True)

    c5, c6 = st.columns((1, 1))
    # H5 - Properties with 3 bathrooms had a MoM growth of 5%
    c5.subheader('H5 - Properties with more than 3 bathrooms had a MoM growth of 5%')
    c5.write('FALSE: There was a monthly increase of 1,64%')

    aux5 = df_h.copy()
    aux5 = aux5.loc[aux5['bathrooms'] > 3].reset_index()
    aux5['date'] = pd.to_datetime(aux5['date'], format='%Y/%m/%d')
    aux5['month'] = pd.DatetimeIndex(aux5['date']).month
    aux5['year'] = pd.DatetimeIndex(aux5['date']).year

    df_h5 = aux5[['price', 'year', 'month']].groupby(['year', 'month']).mean().reset_index()
    df_h5['day'] = 1
    df_h5['year_month'] = pd.to_datetime(df_h5[["year", "month", "day"]])

    fig_h5 = px.bar(df_h5, x='year_month', y='price')
    c5.plotly_chart(fig_h5, use_container_width=True)

    # H6 - Properties with 2 or more floors are 40% more expensive on average
    # than the others
    c6.subheader('H6 - Properties with 2 or more floors are 40% more expensive on average than the others')
    c6.write('TRUE: They are 41.25% more expensive ')

    aux6 = df_h.copy()
    aux6['multi_floors'] = aux6['floors'].apply(lambda x: '>= 2 floors' if x >= 2 else
    '< 2 floors')

    h6 = aux6[['price', 'multi_floors']].groupby('multi_floors').mean().reset_index()

    fig_h6 = px.bar(h6, x='multi_floors', y='price', color='multi_floors')
    c6.plotly_chart(fig_h6, use_container_width=True)

    return None



def final_analysis(df_i):
    total = 991216453.60
    n = 10505

    n_zip = df_i[df_i['zipcode'] == 98052].shape[0]
    n_cond = df_i[df_i['condition'] == 'regular'].shape[0]
    n_grade = df_i[df_i['grade'] == 'average design'].shape[0]
    n_floors = df_i[df_i['floors'] == 1].shape[0]
    n_bed = df_i[df_i['bedrooms'] == 3].shape[0]
    n_bath = df_i[df_i['bathrooms'] == 1].shape[0]
    n_water = df_i[df_i['waterfront'] == 'no'].shape[0]
    n_yr = df_i[df_i['yr_built'] == 1977].shape[0]

    p_zip = df_i['profit'].loc[df_i['zipcode'] == 98052].sum()
    p_cond = df_i['profit'].loc[df_i['condition'] == 'regular'].sum()
    p_grade = df_i['profit'].loc[df_i['grade'] == 'average design'].sum()
    p_floors = df_i['profit'].loc[df_i['floors'] == 1].sum()
    p_bed = df_i['profit'].loc[df_i['bedrooms'] == 3].sum()
    p_bath = df_i['profit'].loc[df_i['bathrooms'] == 1].sum()
    p_water = df_i['profit'].loc[df_i['waterfront'] == 'no'].sum()
    p_yr = df_i['profit'].loc[df_i['yr_built'] == 1977].sum()

    dic = [{'Feature': 'zipcode', 'Status': '98052', 'Count': n_zip, '(%)properties': n_zip * 100 / n, 'Profit': p_zip,
            '(%)Profit': p_zip * 100 / total},
           {'Feature': 'condition', 'Status': 'regular', 'Count': n_cond, '(%)properties': n_cond * 100 / n,
            'Profit': p_cond, '(%)Profit': p_cond * 100 / total},
           {'Feature': 'grade', 'Status': 'average design', 'Count': n_grade, '(%)properties': n_grade * 100 / n,
            'Profit': p_grade, '(%)Profit': p_grade * 100 / total},
           {'Feature': 'floors', 'Status': '1', 'Count': n_floors, '(%)properties': n_floors * 100 / n,
            'Profit': p_floors, '(%)Profit': p_floors * 100 / total},
           {'Feature': 'bedrooms', 'Status': '3', 'Count': n_bed, '(%)properties': n_bed * 100 / n, 'Profit': p_bed,
            '(%)Profit': p_bed * 100 / total},
           {'Feature': 'bathrooms', 'Status': '1', 'Count': n_bath, '(%)properties': n_bath * 100 / n, 'Profit': p_bath,
            '(%)Profit': p_bath * 100 / total},
           {'Feature': 'waterfront', 'Status': 'no', 'Count': n_water, '(%)properties': n_water * 100 / n,
            'Profit': p_water, '(%)Profit': p_water * 100 / total},
           {'Feature': 'year_built', 'Status': '1977', 'Count': n_yr, '(%)properties': n_yr * 100 / n, 'Profit': p_yr,
            '(%)Profit': p_yr * 100 / total}]

    st.subheader('Profit distribution')
    st.write('Profit and properties distribution among features')

    final_table = pd.DataFrame(dic)
    st.dataframe(final_table)
    st.sidebar.write('[github/mariobgallardo - House Rocket](https://github.com/mariobgallardo/house_rocket_project/blob/main/house_rocket_project.ipynb)')

    return None



if __name__ == '__main__':
    # ETL
    # Data extraction
    path = 'properties_sell_list.csv'
    path2 = 'streamlit_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    df_h = get_data(path2)
    geofile = get_geofile( url )
    df_i = get_df_i(data)


    # Transformation
    overview_data( data )
    portfolio_density( data, geofile )
    commercial( data )
    insights(df_i)
    hyphotesys(df_h)
    final_analysis(df_i)


    # Loading















