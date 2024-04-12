#importing libraries needed
import streamlit as st
import xgboost
import time
import sklearn
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle


#setting the page title/configuration
st.set_page_config(page_title="Startups Success Prediction",page_icon="🚀",layout="centered")


#Assigning the available classes for categorical variable(city,p state_code, zip_code) gotten from the dataset. Added others for users who don't have this details
cities = ['San Diego', 'Los Gatos', 'Cupertino', 'San Francisco',
       'Mountain View', 'San Rafael', 'Williamstown', 'Palo Alto',
       'Menlo Park', 'Louisville', 'Brooklyn', 'Denver', 'Vienna',
       'Los Altos', 'Burlingame', 'New York', 'Austin', 'Seattle',
       'Boulder', 'Chicago', 'Berkeley', 'Santa Ana', 'Moffett Field',
       'Durham', 'Pittsburgh', 'San Jose', 'Atlanta', 'Manchester',
       'Sunnyvale', 'Cambridge', 'San Mateo', 'South San Francisco',
       'Boston', 'Waltham', 'Aliso Viejo', 'Kansas City', 'Wilmington',
       'Kirkland', 'Tampa', 'Alameda', 'Bothell', 'Dallas', 'Fremont',
       'Santa Clara', 'Princeton', 'Loveland', 'Kearneysville',
       'Los Angeles', 'Canton', 'Bellevue', 'Washington', 'Somerset',
       'Alpharetta', 'Charlottesville', 'Dulles', 'Bloomfield',
       'Santa Monica', 'Milpitas', 'Raleigh', 'Somerville',
       'Redwood City', 'Timonium', 'Reston', 'Cincinnati', 'Campbell',
       'Sterling', 'Foster City', 'Oakland', 'Petaluma', 'Arlington',
       'Centennial', 'Memphis', 'Plymouth', 'Conshohocken', 'Needham',
       'Newport Beach', 'Longmont', 'Naperville', 'Pasadena',
       'Warrenville', 'Berwyn', 'Morgan Hill', 'Marlborough',
       'Playa Vista', 'Providence', 'Monterey Park', 'Plano',
       'Bingham Farms', 'Philadelphia', 'Freedom', 'Bethesda', 'Portland',
       'Allentown', 'North Billerica', 'Duluth', 'Boxborough',
       'Salt Lake City', 'The Woodlands', 'Burlington', 'Weston',
       'Santa Barbara', 'Columbia', 'SPOKANE', 'Maynard', 'Frederick',
       'West Newfield', 'Long Island City', 'NY', 'Englewood',
       'Solana Beach', 'NW Atlanta', 'Bala Cynwyd', 'San Bruno',
       'Evanston', 'Carlsbad', 'West Hollywood', 'San Franciso',
       'Richardson', 'Zeeland', 'Herndon', 'Paramus', 'Minneapolis',
       'Addison', 'Irvine', 'Woburn', 'New York City', 'Larkspur',
       'Henderson', 'McLean', 'Lowell', 'Woodbury', 'Littleton',
       'Glendale', 'Billerica', 'Broomfield', 'Beverly Hills', 'Hartford',
       'Greenwood Village', 'College Park', 'Napa', 'Itasca', 'Lexington',
       'Calabasas', 'North Reading', 'Lindon', 'Albuquerque', 'Red Bank',
       'Bethlehem', 'Saint Louis', 'Indianapolis', 'New Hope',
       'North Hollywood', 'Waco', 'Carpinteria', 'West Chester',
       'Nashville', 'Bedford', 'Belmont', 'Puyallup', 'Chevy Chase',
       'Chantilly', 'Lake Oswego', 'Tewksbury', 'Dedham', 'Lawrenceville',
       'Jersey City', 'NYC', 'Annapolis', 'Las Vegas', 'Andover',
       'Minnetonka', 'Vancouver', 'Pleasanton', 'El Segundo',
       'Farmington', 'Nashua', 'Saint Paul', 'Sunnnyvale', 'Champaign',
       'Golden Valley', 'Pittsboro', 'Brisbane', 'Westford', 'Emeryville',
       'Hollywood', 'Viena', 'Hillsborough', 'Potomac Falls', 'Tempe',
       'Tualatin', 'Framingham', 'Rye Brook', 'Redmond', 'Yardley',
       'Kenmore', 'Laguna Niguel', 'La Jolla', 'Cleveland', 'Lancaster',
       'Thousand Oaks', 'Provo', 'Columbus', 'Arcadia', 'Yorba Linda',
       'San Carlos', 'Acton', 'Newton', 'Toledo', 'Torrance',
       'Altamonte Springs', 'Westport', 'Chelmsford', 'El Segundo,',
       'Hampton', 'Idaho Falls', 'Scotts Valley', 'Avon', 'Little Rock',
       'Lisle', 'Houston', 'Middleton', 'others']

state_codes = ['CA', 'MA', 'KY', 'NY', 'CO', 'VA', 'TX', 'WA', 'IL', 'NC', 'PA',
       'GA', 'NH', 'MO', 'FL', 'NJ', 'WV', 'MI', 'DC', 'CT', 'MD', 'OH',
       'TN', 'MN', 'RI', 'OR', 'UT', 'ME', 'NV', 'NM', 'IN', 'AZ', 'ID',
       'AR', 'WI', 'others']

zip_codes = ['92101', '95032', '92121', '95014', '94105', '94043', '94041',
       '94901', '1267', '94306', '94025', '40204', '11201', '80202',
       '22182', '94022', '94010', '10004', '94301', '78735', '98122',
       '94103', '80302', '60601', '94303', '94704', '78701', '92705',
       '94035', '98119', '27701', '15219', '10011', '94111', '95134',
       '94107', '10010', '30303', '3101', '98102', '94089', '94085',
       '100011', '2139', '94607', '94404', '94080', '2210', '78731',
       '2451', '95128', '92656', '64106', '1887', '98033-6314', '33609',
       '02111-1720', '94501', '98011', '94403-1855', '94104', '94087',
       '33626', '75001', '94538', '95051', '2138', '8540', '80538',
       '10012', '25430', '90025', '48187', '98004', '95054', '20007',
       '94110', '10173', '60606', '8873', '94710', '30009', '98104',
       '22902', '20166', '60607', '6002', '60401', '95035-6261', '27606',
       '10001', '2143', '94065', '10003', '21093', '15213', '20191',
       '10019', '45202', '30305', '2142', '94403', '95008', '94107-4132',
       '10014', '30308', '94609', '94954', '2109', '22201', '98101',
       '80112', '60631', '94114', '38103-4717', '10013', '94010-4031',
       '10016', '55441', '78730', '94108', '19428', '2494', '92660',
       '80501', '94063', '60563', '91101', '95035', '2118', '15218',
       '90405', '60555', '27607', '19312', '94086', '95037', '78729',
       '01752-4603', '94115', '10154', '90094', '2906', '20005',
       '94043-1107', '91754', '75023', '91105', '48025', '60657', '10036',
       '95019-2901', '2129', '98109', '20814', '97209', '18106',
       '01862-2000', '90017', '30097', '2903', '94102', '1719', '84124',
       '77380', '1803', '98103', '94401', '33327', '94304', '93101',
       '90034', '90401', '94040', 'Maryland 21045', '99202', '1754',
       '21703', '4095', '11101', '98007', '90275', '30309', '19004',
       '94066', '60201', '92010', '90069', '94017', '78759', '92130',
       '75081', '49464', '20170', '7652', '55311', '27713', '10005',
       '98033', '92612', '1801', '94939', '90046', '2114', '92618',
       '90404', '89052', '2116', '22102', '1851', '11797', '1460',
       '90032', '91203', '60610', '10017', 'CA 94105', '1821', '95030',
       '80021', '97214', '2111', '90211', '6103', '80111', '20742',
       '90015', '94558', '10119', '60143', '2421', '95054-2913', '33618',
       '94402', '91302', '1864', '90049', '2110', '75082', '84042',
       '90064', '87102', '7701', '18015', '63043', '10018', '19104',
       '90024', '46204', '18938', '80155', '90039', '91601', '76710',
       '93013', '60654', '19380', '37201', '1730', '94002', '37212',
       '98371', '20815', '98005', '80027', '95008-2069', '40507', '22031',
       '97035', '1876', '78746', '2026', '30043', '7302', '21401',
       '95138', '20009', '89101', '1810', '94024', '55343', '90036',
       '98665', '94566', '90245', '6032', '94133', '90038', '33635',
       '1752', '3062', '98121', '55113', '19103', '92128', '61820',
       '55416', '27312', '2140', '94005', '1886', '27518', '2453',
       '11217', '11222', '78702', '95110', '94608', '95112', '10007',
       '2452', '94555-3619', '30350', '92606', '90028', '90301', '8844',
       '20165', '93105', '85281', '97062', '95035-5444', '1702', '10573',
       '98052', '11211', '19067', '92009', '94109', '94903', '97201',
       '14217', '92677', '94040-1309', '92037', '78703', '94588',
       '92130-2042', '44131', '46250', '17603', '95131', '91360', '84087',
       '95032-5405', '43207', '97204', '80205', '91006', '98006',
       '94063-2026', '92887', '94070', '941017', '1720', '75244', '2466',
       '85284', '43607', '92008', '10038', '90501', '90291', '32714',
       '6880', '1824', '60647', 'CA 90245', '8827', '83402', '94040-2573',
       '95066', '81620', '97224', '94301-1705', '72201', '60188', '77046',
       '94410', '77027', '53562', '98021', 'others']

categories = ['music', 'enterprise', 'web', 'software', 'games_video',
       'network_hosting', 'finance', 'mobile', 'education',
       'public_relations', 'security', 'other', 'photo_video', 'hardware',
       'ecommerce', 'advertising', 'travel', 'fashion', 'analytics',
       'consulting', 'biotech', 'cleantech', 'search', 'semiconductor',
       'social', 'medical', 'automotive', 'messaging', 'manufacturing',
       'hospitality', 'news', 'transportation', 'sports', 'real_estate',
       'health', 'others']

count = 1


#creating a method to represent the first page i.e Home Page
def Home_Page():

    
    #placing the heading 'Startup Success Prediction' using css
    st.write(
                    """
                    <div style="background-color: #4682B4; border-radius: 20px; padding: 5px; color: white; font-weight: bold; text-align: center; font-size: 46px;">
                        Startups Success Prediction
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
    #creating interface of home page
    st.write("\n")
    st.write("## VC being skeptical about a statup?")
    st.write("")
    st.write('Check the probability of success here!  \n*This system predict the probabilty of success with just a little information*' )
    st.image('How-to-Measure-Startup-Success-Main.jpg')
    st.write("  \n")
    st.write('## Enter Features and get Predictions')
    st.write("  \n")

    
    #importing cleaned dataset exported from machine notebook for encoding and preprocessing user input
    df = pd.read_csv('startup.csv')

     #creating object for label encoder
    le = LabelEncoder()    


    #creating empty dictionary to store user input
    test = {}

    
    #method created to preprocess categorical inputs by encoding and storing to created dictionary
    #this methode insert the model value (most common category) if user click 'others'
    #this method increament a variable called number of unknown if the user enters 'others' to track the unknown entries
    def categorical(column, column_str):

        if column == 'others':
            test[column_str] = le.fit(df[column_str]).transform([df[column_str].mode()[0]])[0]
            global count 
            count+=1
        else:
            test[column_str] = (le.fit(df[column_str]).transform([column]))[0]

    #this method writes continous entries to the test dictionary and take care of no entry
    def continous(column, column_str):
        if column is None:
            test[column_str] = df[column_str].mean()
            global count
            count+=1
        else:
            test[column_str] = column


    #method to encode date entries and create features for day, month and year
    def dates(column, column_str):
        test[column_str+'_year'] = column.year
        test[column_str+'_day'] = column.day
        test[column_str+'_month'] = column.month
            
    #method to encode binnary column, assumes negative if no entry i.e user entered 'do not know
    def binary(column, column_str):
        if column=="don't know":
            test[column_str] = 0
            global count
            count+=1
        elif column == 'yes':
            test[column_str] = 1
        else:
            test[column_str] = 0

    #method to calculate age using given dates
    def findage(date1, date2, key):
        delta = date2 - date1
        years_difference = delta.days / 365.25
        test[key] = years_difference


    #this method assumes the user didn't enter date if the returned date is today's date
    #if the user did not enter a date, the method sets the most common year, day and month in that date column as the date
    def set_unknown_dates(column, column_str):
        if column == datetime.today().date():
            column = column.replace(year = df[column_str+'_year'].mode()[0], day = df[column_str+'_day'].mode()[0], month = df[column_str+'_month'].mode()[0] )
            global count
            count+=1

    






    #calling the appropraite function to preprocess user entered data
    def make_prediction():
        set_unknown_dates(founded_at, 'founded_at')
        set_unknown_dates(first_funding_at, 'first_funding_at')
        set_unknown_dates(last_funding_at, 'last_funding_at')
        categorical(state_code, 'state_code')
        continous(latitude, 'latitude')
        continous(longitude, 'longitude')
        categorical(zip_code, 'zip_code')
        categorical(city, 'city')
        findage(founded_at, first_funding_at, 'age_first_funding_year')
        findage(founded_at, last_funding_at, 'age_last_funding_year')
        findage(founded_at, first_milestone_at, 'age_first_milestone_year')
        findage(founded_at, last_milestone_at, 'age_last_milestone_year')
        continous(relationships, 'relationships')
        continous(funding_rounds, 'funding_rounds')
        continous(funding_total_usd, 'funding_total_usd')
        continous(milestones, 'milestones')
        categorical(category_code, 'category_code')
        continous(avg_participants, 'avg_participants')
        binary(is_top_500, 'is_top500')
        dates(founded_at, 'founded_at')
        dates(first_funding_at, 'first_funding_at')
        dates(last_funding_at, 'last_funding_at')
        
        #converting test to dataframe 
        test_df = pd.DataFrame(test, index = ['values'])
        test_df['funding_total_usd'] = pd.to_numeric(test_df['funding_total_usd'], errors='coerce')
        test_df['avg_participants'] = pd.to_numeric(test_df['avg_participants'], errors='coerce')

        #importing model
        with open('model', 'rb') as f:
            model = pickle.load(f)

        #making prediction and printing output only if the number of unknown from user is less than or equal to 8
        if count <=8:
            prediction = model.predict_proba(test_df)
            
                   
            
            st.write('probability of success: ', prediction[0,0], 'probability of failure: ', prediction[0,1])
            

            if prediction[0,0]<prediction[0,1]:
                st.write(
                """
                <div style="background-color: #EOFFFF; border-radius: 10px; padding: 5px; color: blue; font-weight: bold; text-align: center; font-size: 16px;">
                    Statup will succeed
                </div>
                """,
                unsafe_allow_html=True
                )
            else:
                st.write(
                """
                <div style="background-color: #EOFFFF; border-radius: 0px; padding: 5px; color: red; font-weight: bold; text-align: center; font-size: 16px;">
                    Startup will fail
                </div>
                """,
                unsafe_allow_html=True
                )
        else:
            st.write(
                """
                <div style="background-color: #EOFFFF; border-radius: 0px; padding: 5px; color: red; font-weight: italize; text-align: center; font-size: 16px;">
                    error! enter more features
                </div>
                """,
                unsafe_allow_html=True
                )
        




    #divide the column into 3
    col1, col2= st.columns(2)

    #getting input from user from column 1 and column 2
    with col1:
        state_code = st.selectbox("select state code of the location", state_codes,)
        latitude = st.number_input("Enter the Latitude",)
        longitude = st.number_input("Enter the Longitude")
        zip_code = st.selectbox('enter the zip code', zip_codes)
        city = st.selectbox('select the city where the startup is located', cities)
        category_code = st.selectbox('select the category of the startup', categories)
        founded_at = st.date_input('founding date', min_value= datetime(1980, 1, 1))
        first_funding_at = st.date_input('date of first funding', min_value= datetime(1980, 1, 1))
        last_funding_at = st.date_input('date of last funding', min_value= datetime(1980, 1, 1))
                
    with col2:
        first_milestone_at = st.date_input('date of first milestone', min_value= datetime(1980, 1, 1))
        last_milestone_at = st.date_input('date of last milestone', min_value= datetime(1980, 1, 1))
        relationships = st.number_input('enter the numbers of relationships', 0, 100 )
        funding_rounds = st.slider('select the number of funding rounds', 0, 10)
        funding_total_usd = st.text_input('enter the total funding amount in usd')
        milestones = st.slider('enter the number of milestones', 0, 10)
        avg_participants = st.text_input('enter the average number of participant per funding rounds')
        is_top_500 = st.radio('is startup among top 500?', ['yes', 'no', "don't know"])
    

    st.write(" ")
    st.write(" ")
    st.write(" ") 
    
    #placing predict button
    predict = st.button('Predict', use_container_width=True, help ='click here to get prediction')
    
    
    #if predict button is clicked show progress bar for 5 minute and call make_prediction() method to make prediction
    if predict:
        bar = st.progress(0)
        for i in range(5):
            bar.progress((i+1)*20)
            time.sleep(1)
        make_prediction()
#method for about project page     
def About_project():
    #Title
    st.markdown("<h1 style='text-align:center;'>Startup Success Prediction</h1>",unsafe_allow_html=True)

    #Problem Statement Section
    st.markdown("<div style='background-color:#EC7063; padding:10px; border-radius:25px; text-align:center;'><b>Problem Description</b></div>",unsafe_allow_html=True)

    st.write("")

    st.markdown("<div style='background-color:#CACFD2; padding:10px; border-radius:15px; text-align:left;'>Startups face significant challenges in their journey to success, with factors ranging from market demand and competition to internal team dynamics influencing their outcomes. In this project, we aim to develop a machine learning model that predicts the likelihood of success for startups based on various features such as funding, team size, industry sector, and geographical location. The primary objective is to provide entrepreneurs and investors with a tool that can assess the potential success of a startup at an early stage, enabling better decision-making and resource allocation. By leveraging historical data on startup performance, our model will offer valuable insights into the key drivers of success and help stakeholders identify promising investment opportunities or areas for strategic focus.'</div>",unsafe_allow_html=True)

    st.write("")
    
    #How to use Application
    st.markdown("<div style='background-color:#EC7063; padding:10px; border-radius:25px; text-align:center;'><b>How to use Application</b></div>",unsafe_allow_html=True)

    st.write("")

    st.markdown("<div style='background-color:#CACFD2; padding:10px; border-radius:15px; text-align:left;'>'This application gives accurate prediction of the probability of a startup succeeding versus failing. The model was train on a publicly available dataset gotten from kaggle. this was a labeled crunch base data and the model achieved an accuracy of 81%.  \nTo use this application provide the requested features for the startup. The model will produce an output if user can provide more than half of the amount of features needed i.e 9 and above. Provided user can't make available this required amount of features then the application concludes user don't have enough information to make prediction on the startup. Note: providing an incomplete feature set for prediction reduces the performance of the application i.e reduces the accuracy.  \nClicking on predict starts the model to make prediction on the data make available. it advicable to provide complete data for accurate prediction.'</div>",unsafe_allow_html=True)
    
    st.write("")

    #Dataset Description Section
    st.markdown("<div style='background-color:#EC7063; padding:10px; border-radius:25px; text-align:center;'><b>Dataset Description</b></div>",unsafe_allow_html=True)

    st.write("")

    st.markdown("<div style='background-color:#CACFD2; padding:10px; border-radius:15px; text-align:left;'>This dataset is related to startups, it was scraped from Crunchbase. It contains information about startup locations, founding details, funding rounds, categories, and their success status. It could be a valuable resource for analyzing factors influencing startup success.'</div>",unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        st.write("")
        st.link_button(label="Github repo",url="https://github.com/AgabaEmbedded/ML-Hackathon",use_container_width=True)
        st.write("")
    with col5:
        st.write("")
        st.link_button(label="View Dataset",url="https://www.kaggle.com/datasets/manishkc06/startup-success-prediction",use_container_width=True)
    with col4:
        st.write("")
        st.link_button(label="Notebook",url="https://www.kaggle.com/code/sundayabraham/startups-success-prediction",use_container_width=True)
        st.write("")
        
    st.write("")

    #Detail description of dataset
    st.markdown("<div style='background-color:#EC7063; padding:10px; border-radius:25px; text-align:center;'><b>Detailed Description of Dataset</b></div>",unsafe_allow_html=True)

    st.markdown("")

    with st.expander('View Details'):

        st.write("# description of each Column")
        st.markdown('**Unnamed: 0:** This column name often appears in datasets due to the way they are exported or created. It does not contain meaningful data and was dropped during data cleaning.')
        st.markdown('\n**state_code:** Two-letter code representing the state where the startup is located (e.g., CA for California, NY for New York).')
        st.markdown('**latitude, longitude:** Geographical coordinates of the location of the startup.')
        st.markdown('**zip_code:** Postal code of the location of the startup.')
        st.markdown('**id:** Unique identifier for each startup in the dataset.')
        st.markdown('**city:** Name of the city where the startup is located.')
        st.markdown('**Unnamed: 6:** Similar to "Unnamed: 0," this does not hold valuable data and was dropped.')
        st.markdown('**name:** Name of the startup.')
        st.markdown('**labels:** Labels or tags associated with the startup this is a duplicate of the target class.')
        st.markdown('**founded_at:** Date or year when the startup was founded.')
        st.markdown('**closed_at:** Date or year when the startup ceased operations (if applicable).')
        st.markdown("**first_funding_at:** Date or year of the startup's first funding round (if applicable).")
        st.markdown("**last_funding_at:** Date or year of the startup's most recent funding round (if applicable).")
        st.markdown('**age_first_funding_year:** Calculated age of the startup when it received its first funding (years since founding).')
        st.markdown('**age_last_funding_year:** Calculated age of the startup at its most recent funding (years since founding).')
        st.markdown('**age_first_milestone_year:** time elapsed between founding and achieving a significant milestone.')
        st.markdown('**age_last_milestone_year:** the time elapsed until the most recent milestone.')
        st.markdown("**relationships:** details about the startup's connections to other entities (investors, partners, etc.).")
        st.markdown('**funding_rounds:** Total number of funding rounds the startup has received.')
        st.markdown('**funding_total_usd:** Total amount of funding raised by the startup in USD (across all rounds).')
        st.markdown('**milestones:** List of milestones achieved by the startup.')
        st.markdown('**state_code.1 (duplicate?):** This is a duplicate of the original "state_code" column and was be dropped.')
        st.markdown("**is_CA, is_NY, is_MA, is_TX, is_otherstate:** Binary indicators for specific states (e.g., California, New York, Massachusetts, Texas, Other).")
        st.markdown("**category_code:** Code representing the startup's industry category (e.g., software, healthcare).")
        st.markdown("**is_software, is_web, is_mobile, is_enterprise, is_advertising, is_gamesvideo, is_ecommerce, is_biotech, is_consulting, is_othercategory:** Binary indicators for specific categories or industry sectors.")
        st.markdown('**object_id: Identifier for the startup (similar to id column).')
        st.markdown('**has_VC, has_angel to has_roundD:** Likely binary indicators signifying whether the startup received funding from specific sources (VC = Venture Capital, Angel Investors, funding rounds A-D).')
        st.markdown("**avg_participants:** Average number of participants (investors) in the startup's funding rounds.")
        st.markdown("**is_top500:** Binary indicator if the startup is among the top 500 companies")
        st.markdown('**status:** Status of the startup (e.g., closed, acquired) where acquired indicate both acquired or operating. this is the target class')
    


#creat sidebar for pages navigation
st.sidebar.markdown("<h1 style='text-align:center;'>Hi 👋, Welcome to Start Ups all in one App</h1>",unsafe_allow_html=True)    

#place sidebar logo
st.sidebar.image("logo.png",caption="Startup Success Prediction",use_column_width=True)

st.sidebar.markdown("<div style='text-align:center; font-size:x-large;'><b>Select any Page</b></div>",unsafe_allow_html=True) 

pages = st.sidebar.selectbox(label="",options=["Home Page","About Project"],index=0)

#pages navigation
if pages == "Home Page":
    Home_Page()
else:
    About_project()
