import xgboost as xgb
import FreeSimpleGUI as sg
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')

x_values = []
y_values = []
def draw_plot(x_values, y_values):
    # Creating the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, color='g')  # Plot as a continuous line
    plt.ylim(0, 100)  # Adjust the y-axis limits for life expectancy range (adjust as necessary)

    # Labels and title for clarity
    plt.xlabel('Prediction Index / Time Step')
    plt.ylabel('Life Expectancy')
    plt.title('Continuous Life Expectancy Predictions')

    # Show the updated plot
    plt.draw()
    plt.pause(0.001)  # This pauses to update the plot in real-time


def predict(features):
    # Loading the xgboost model from JSON
    loaded_model = xgb.XGBRegressor()
    loaded_model.load_model('C:\\Users\\Administrator\\Desktop\\Big Data\\life_expectancy\\xgboost_model.json')
    feature_names = loaded_model.get_booster().feature_names

    df = pd.DataFrame([features], columns = feature_names)
    predictions = loaded_model.predict(df)

    return predictions

# All the stuff inside your window.
layout = [  [sg.Text('Fill in the features to get the predicted life expectancy. Be aware of the units of each variable!')],


            [sg.Text("The predicted life expectancy for the selected features is:  ", key='life_expectancy')],
            [sg.Text('Infant deaths:'), sg.InputText(key='infant_deaths',default_text="11.9")],
            [sg.Text('Under 5 Year Deaths:'), sg.Slider(range=(0, 1000), default_value=14, orientation='h',key='under_5_year')],
            [sg.Text('Adult Mortality:'), sg.InputText(key='adult_mortality',default_text="151.8615")],
            [sg.Text('Alcohol Consumption:'), sg.Slider(range=(0, 100), default_value=12.38, orientation='h',key='alcohol_consumption')],
            [sg.Text('Hepatitis B:'), sg.Slider(range=(0, 100), default_value=97, orientation='h',key='hepatitis_b')],
            [sg.Text('Measles:'), sg.InputText(key='measles',default_text="95")],
            [sg.Text('BMI:'), sg.InputText(key='bmi',default_text="25.8")],
            [sg.Text('Polio:'), sg.InputText(key='polio',default_text="95")],
            [sg.Text('Diphtheria:'), sg.InputText(key='diptheria',default_text="96")],
            [sg.Text('Incidents HIV:'), sg.Slider(range=(0, 100), default_value=0.04, orientation='h',key="hiv_incident")],
            [sg.Text('GDP/Capita:'), sg.Slider(range=(0, 100000), default_value=8280, orientation='h',key="gdp_per_capita")],
            [sg.Text('Thinness_ten_nineteen_years:'), sg.Slider(range=(0, 100), default_value=3.1, orientation='h',key="thinnes_10-19_year")],
            [sg.Text('Thinness_five_nine_years:'), sg.InputText(key="thinnes_five_nine_year",default_text="3.5")],
            [sg.Text('Schooling:'), sg.InputText(key = "schooling",default_text="10.6")],
            [sg.Text('Economy_status Developed:'), sg.InputText(key = "developed",default_text="1")],
            [sg.Text('Economy_status Developing:'), sg.InputText(key = "developing",default_text="0")],
            [sg.Text('Smoking Death Rate:'), sg.Slider(range=(0, 10000), default_value=140.9, orientation='h',key = "smoking_death_rate")],
            [sg.Button('Predict'), sg.Button('Cancel')] ]

# Create the Window
window = sg.Window('Window Title', layout)

delay = x = lastx = lasty = 0
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break



    infant_deaths = float(values['infant_deaths']) if values['infant_deaths'] else 0.0
    under_5_year = float(values['under_5_year']) if values['under_5_year'] else 0.0
    adult_mortality = float(values['adult_mortality']) if values['adult_mortality'] else 0.0
    alcohol_consumption = float(values['alcohol_consumption']) if values['alcohol_consumption'] else 0.0
    hepatitis_b = float(values['hepatitis_b']) if values['hepatitis_b'] else 0.0
    measles = float(values['measles']) if values['measles'] else 0.0
    bmi = float(values['bmi']) if values['bmi'] else 0.0
    polio = float(values['polio']) if values['polio'] else 0.0
    diptheria = float(values['diptheria']) if values['diptheria'] else 0.0
    hiv_incident = float(values['hiv_incident']) if values['hiv_incident'] else 0.0
    gdp_per_capita = float(values['gdp_per_capita']) if values['gdp_per_capita'] else 0.0
    thinnes_10_19_year = float(values['thinnes_10-19_year']) if values['thinnes_10-19_year'] else 0.0
    thinnes_five_nine_year = float(values['thinnes_five_nine_year']) if values['thinnes_five_nine_year'] else 0.0
    schooling = float(values['schooling']) if values['schooling'] else 0.0
    developed = float(values['developed']) if values['developed'] else 0.0
    developing = float(values['developing']) if values['developing'] else 0.0
    smoking_death_rate = float(values['smoking_death_rate']) if values['smoking_death_rate'] else 0.0

    predicates =[0]

    if event == 'Predict':
        y= predict([infant_deaths, under_5_year, adult_mortality, alcohol_consumption, hepatitis_b, measles, bmi, polio,
                 diptheria, hiv_incident, gdp_per_capita, thinnes_10_19_year, thinnes_five_nine_year, schooling,
                 developed, developing, smoking_death_rate])
        window["life_expectancy"].update(y)

        x_values.append(len(x_values))  # Each new prediction gets the next index (time step)
        y_values.append(y[0])  # The predicted life expectancy value
        draw_plot(x_values, y_values)

window.close()