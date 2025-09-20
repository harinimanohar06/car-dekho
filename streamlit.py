import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load scaler
with open("c:/Users/HARINI/OneDrive/Desktop/streamlit/fuel_encoder.pkl", 'rb') as file:
   fuel= pickle.load(file)
with open('c:/Users/HARINI/OneDrive/Desktop/streamlit/body_encoder.pkl', 'rb') as file:
   body= pickle.load(file)
with open('c:/Users/HARINI/OneDrive/Desktop/streamlit/transmission_encoder.pkl', 'rb') as file:
   transmission= pickle.load(file)
with open('c:/Users/HARINI/OneDrive/Desktop/streamlit/label_manufacturer_encoder.pkl', 'rb') as file:
   manufacturer_encoder= pickle.load(file)
with open('c:/Users/HARINI/OneDrive/Desktop/streamlit/car_model_encoder.pkl', 'rb') as file:
   model_encoder= pickle.load(file)
with open('c:/Users/HARINI/OneDrive/Desktop/streamlit/insurance_encoder.pkl', 'rb') as file:
   insurance_encoder= pickle.load(file)


with open("c:/Users/HARINI/OneDrive/Desktop/streamlit/ohe_encoder.pkl", 'rb') as file:
   ohe_encoder= pickle.load( file)


# Clean file path (no leading space!)
model_path = 'c:/Users/HARINI/OneDrive/Desktop/streamlit/rf_model.pkl'

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please check the path.")
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🚗 **Car Price Prediction App**")

fuel_type =st.selectbox("Select a fuel type",("Petrol","Diesel",'Lpg' ,'Cng' ,'Electric'))
fuel1= fuel.transform([fuel_type])
body_type =st.selectbox("Select a body type",('Hatchback' ,'SUV','Sedan', 'MUV', 'Coupe', 'Minivans', 'Pickup Trucks',
 'Convertibles' ,'Hybrids' ,'Wagon' ))
bodytype1= body.transform([body_type])
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, step=1000)
st.write("KMs Driven:", kms_driven)
transmission_type =st.selectbox("Select a transmission type",("Automatic","Manual"))
transmission1= transmission.transform([transmission_type])
ownerNo = st.number_input("Number of Previous Owners", min_value=0, max_value=5, step=1)
st.write("Previous Owners:", ownerNo)
Manufacturer=st.selectbox("Select a manufacture",('Maruti','Ford','Tata','Hyundai','Jeep','Datsun', 'Honda' ,'Mahindra',
 'Mercedes-Benz' ,'BMW' ,'Renault', 'Audi' ,'Toyota','Mini', 'Kia' ,'Skoda',
 'Volkswagen', 'Volvo' ,'MG', 'Nissan', 'Fiat', 'Mahindra Ssangyong',
 'Mitsubishi' ,'Jaguar' ,'Land Rover' ,'Chevrolet', 'Citroen' ,'Opel',
 'Mahindra Renault', 'Isuzu', 'Lexus', 'Porsche', 'Hindustan Motors'))
Manufacturer1= manufacturer_encoder.transform([[Manufacturer]])
Model =st.selectbox("Select a model type",('Maruti Celerio' ,'Ford Ecosport' ,'Tata Tiago' ,'Hyundai Xcent',
 'Maruti SX4 S Cross' ,'Jeep Compass' ,'Datsun GO' ,'Hyundai Venue',
 'Maruti Ciaz', 'Maruti Baleno' ,'Hyundai Grand i10', 'Honda Jazz',
 'Mahindra XUV500' ,'Mercedes-Benz GLA', 'Hyundai i20', 'Tata Nexon',
 'Honda City' ,'BMW 5 Series', 'Maruti Swift' ,'Renault Duster',
 'Mercedes-Benz S-Class', 'Hyundai Santro' ,'Hyundai Santro Xing',
 'Mercedes-Benz E-Class' ,'Audi A4', 'Maruti Wagon R' ,'Maruti Ertiga',
 'Mercedes-Benz C-Class', 'Toyota Fortuner', 'Hyundai Elantra' ,'Audi A6',
 'Maruti Alto 800' ,'Mahindra Scorpio' ,'Mini 3 DOOR', 'Kia Seltos',
 'Maruti Alto' ,'Mercedes-Benz GL-Class', 'Tata New Safari', 'Audi Q7',
 'Renault KWID' ,'Hyundai Getz' ,'Skoda Rapid' ,'Hyundai Creta',
 'Tata Harrier' ,'BMW 3 Series GT', 'Renault Lodgy', 'Skoda Octavia',
 'Maruti Ritz' ,'Volkswagen Polo' ,'Mahindra KUV 100', 'BMW X3', 'Hyundai i10',
 'Volvo S60' ,'Mahindra XUV300', 'MG Hector Plus' ,'Honda Brio','Maruti Alto K10', 'Renault Kiger' ,'Hyundai EON' ,'Volkswagen Vento',
 'Toyota Yaris', 'MG Hector' ,'Hyundai Alcazar' ,'Volkswagen T-Roc',
 'BMW 3 Series' ,'Skoda Superb' ,'Audi Q5', 'Ford Endeavour' ,'Ford Figo',
 'Maruti Ignis' ,'Renault Triber', 'BMW X5', 'Hyundai Tucson' ,'Hyundai Verna',
 'Mercedes-Benz GLC' ,'Nissan Terrano', 'Honda CR-V',
 'Mercedes-Benz A-Class Limousine', 'Toyota Innova' ,'Hyundai Santa Fe',
 'BMW 6 Series', 'Maruti Baleno RS' ,'Renault Captur' ,'Maruti Vitara Brezza',
 'Maruti Swift Dzire' ,'Fiat Linea' ,'Hyundai i20 Active' ,'Honda WR-V',
 'Mahindra Ssangyong Rexton', 'Toyota Corolla Altis' ,'Ford Ikon',
 'Mitsubishi Cedia' ,'Jaguar XF' ,'Audi A3', 'Skoda Kushaq',
 'Volkswagen Taigun', 'MG Astor', 'Hyundai Accent' ,'Mercedes-Benz B Class',
 'Kia Carnival' ,'Skoda Laura' ,'BMW X4', 'Mini Cooper',
 'Land Rover Discovery Sport' ,'Volvo XC40' ,'Kia Sonet' ,'Mahindra Verito',
 'Maruti S-Presso' ,'Volkswagen Jetta' ,'Datsun RediGO' ,'Ford Aspire',
 'Ford Freestyle' ,'Audi Q3' ,'Tata Tigor' ,'Jaguar F-Pace',
 'Mercedes-Benz A Class' ,'Toyota Glanza', 'Nissan Magnite',
 'Tata Safari Storme' ,'Maruti Celerio X' ,'Mercedes-Benz M-Class',
 'Mercedes-Benz GLE', 'Toyota Urban cruiser' ,'Mahindra Thar',
 'Mercedes-Benz CLA' ,'MG Comet EV' ,'Maruti Omni', 'Volkswagen Tiguan',
 'Tata Altroz' ,'Tata Nexon EV Max', 'Tata Indica V2' ,'Toyota Innova Crysta',
 'Volkswagen Ameo' ,'Tata Nexon EV Prime', 'BMW X1', 'Chevrolet Cruze',
 'Toyota Camry' ,'Fiat Punto Abarth' ,'Mahindra TUV 300', 'Chevrolet Beat',
 'Maruti Eeco', 'Maruti 1000' ,'Citroen C5 Aircross', 'Mahindra XUV700',
 'Hyundai Grand i10 Nios', 'Maruti Zen' ,'Mahindra Quanto',
 'Land Rover Freelander 2', 'OpelCorsa' ,'Mahindra Xylo' ,'Tata Zest',
 'Honda New Accord' ,'Skoda Yeti' ,'Maruti SX4', 'Jaguar XE',
 'Chevrolet Spark' ,'Hyundai i20 N Line' ,'Chevrolet Tavera', 'BMW X7',
 'Mahindra Renault Logan' ,'Mahindra e2o Plus' ,'Citroen C3' ,'Tata Nano',
 'Honda Amaze', 'Mahindra Bolero Power Plus' ,'Tata Manza' ,'Maruti Esteem',
 'Tata Hexa' ,'Nissan Micra Active' ,'Mitsubishi Lancer', 'Ford Fiesta',
 'Mahindra Bolero Camper' ,'Fiat Punto' ,'Kia Carens' ,'Chevrolet Enjoy',
 'Volkswagen Tiguan Allspace', 'Skoda Slavia' ,'Mahindra Marazzo',
 'Tata Indigo', 'Jaguar XJ' ,'Skoda Fabia' ,'Tata Sumo' ,'Ford Mondeo',
 'Nissan Sunny' ,'Fiat Palio' ,'Toyota Etios' ,'Maruti Estilo',
 'Mahindra Bolero' ,'Jeep Meridian' ,'BMW 1 Series' ,'Volvo XC 90',
 'Audi A3 cabriolet' ,'MG Gloster', 'Land Rover Range Rover Sport',
 'Nissan Micra', 'Fiat Punto EVO' ,'Mini Cooper Countryman',
 'Renault Fluence' ,'Maruti A-Star' ,'Tata Nexon EV' ,'Chevrolet Sail',
 'BMW 7 Series' ,'Maruti XL6' ,'Hyundai Sonata' ,'Honda Civic',
 'Maruti Ertiga Tour' ,'Mercedes-Benz GLS' ,'Isuzu MU 7' ,'Maruti 800',
 'Hyundai Aura' ,'BMW 3 Series Gran Limousine' ,'Volvo S90' ,'Tata Indica',
 'Tata Punch' ,'Honda BR-V', 'Mahindra Scorpio N', 'Skoda Kodiaq',
 'Tata Tiago NRG', 'Datsun GO Plus' ,'BMW 2 Series',
 'Maruti Wagon R Stingray' ,'Mini 5 DOOR' ,'Fiat Grande Punto',
 'Mahindra KUV 100 NXT' ,'Mercedes-Benz GLA Class', 'Chevrolet Aveo',
 'Land Rover Range Rover Velar', 'Toyota Hyryder', 'Maruti Zen Estilo',
 'Toyota Etios Liva' ,'Land Rover Range Rover Evoque' ,'Maruti Versa',
 'Isuzu MU-X', 'Fiat Punto Pure' ,'Honda Mobilio' ,'Chevrolet Optra',
 'Volvo S 80', 'Mitsubishi Pajero' ,'Audi A8' ,'Volvo XC60',
 'Mercedes-Benz AMG GLA 35' ,'Mercedes-Benz AMG A 35', 'Volkswagen Virtus',
 'Land Rover Discovery' ,'Lexus ES' ,'Audi Q2', 'Nissan Kicks',
 'Mahindra TUV 300 Plus', 'Maruti Brezza' ,'Jeep Wrangler',
 'Toyota Etios Cross', 'Land Rover Defender' ,'Mercedes-Benz GLC Coupe',
 'Lexus RX' ,'Mitsubishi Outlander' ,'Mercedes-Benz CLS-Class',
 'Maruti Jimny' ,'Mini Cooper Clubman', 'Porsche Cayenne',
 'Maruti Swift Dzire Tour' ,'Mercedes-Benz G' ,'Mini Cooper Convertible',
 'Mercedes-Benz SLC' ,'Isuzu D-Max' ,'Maruti Grand Vitara',
 'Ford Fiesta Classic', 'Maruti FRONX' ,'Mahindra Alturas G4',
 'Volvo S60 Cross Country', 'Jeep Compass Trailhawk' ,'Renault Scala',
 'Tata Sumo Victa', 'Porsche Macan' ,'Volvo V40', 'Porsche Panamera',
 'Ambassador' ,'Mercedes-Benz AMG GT', 'Audi S5 Sportback' ,'Renault Pulse',
 'Mahindra Bolero Neo', 'Jaguar F-TYPE' ,'Tata Tigor EV','Toyota Fortuner Legender' ,'Hyundai Xcent Prime', 'Mahindra Jeep',
 'Toyota Qualis' ,'Volkswagen Passat', 'Maruti Gypsy',
 'Land Rover Range Rover' ,'Fiat Avventura' ,'Honda City Hybrid' ,'Tata Aria','Tata Bolt', 'MG ZS EV' ,'Mahindra E Verito' ,'Mercedes-Benz EQC','Fiat Abarth Avventura' ,'Hindustan Motors Contessa' ,'Tata Yodha Pickup','Tata Indigo Marina' ,'Chevrolet Captiva','Mahindra Bolero Pik Up Extra Long' ,'Toyota Corolla','Mercedes-Benz AMG GLC 43' ,'Chevrolet Aveo U-VA' ,'Hyundai Kona','Porsche 911' ,'Volkswagen CrossPolo'))
model1 = model_encoder.transform([Model])
mileage = st.number_input("Select mileage (in kmpl)", min_value=5, max_value=40, value=15)
st.write("Selected mileage:", mileage)
max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=1000.0, value=100.0, step=10.0)
st.write("select max_power",max_power)
seats = st.selectbox('Select the number of seats:', options=[4, 5,])
st.write(f"You selected: {seats} seats")
max_power= 0.00  
result = max_power + seats * 0.03
insurance_validity=st.selectbox("Select a insurance",('Third Party insurance' ,'Comprehensive' ,'Third Party', 'Zero Dep', 'Petrol','2', 'Diesel', '1' ,'Not Available', 'Electric'))
insurance1= insurance_encoder.transform([insurance_validity]).flatten()
#st.write(insurance1)
engine_cc = st.number_input("Engine Displacement (cc)", min_value=500, max_value=5000, step=100, value=1200)
st.write("Engine CC:", engine_cc)
car_age = st.number_input("Car Age (in years)", min_value=0, max_value=30,  step=1, value=5  )
st.write("Car Age:", car_age)
city = st.selectbox("Select City",('banglore','chennai', 'delhi' ,'kolkata' ,'hyderabad' ,'jaipur'))
city_encoded = ohe_encoder.transform([[city]])
# Proper input construction
fuel1 = fuel.transform([fuel_type]).flatten()  # 1D
bodytype1 = np.array([body.transform([body_type])[0]])  # scalar to 1D
kms_driven = np.array([kms_driven])  # scalar to 1D
transmission1 = transmission.transform([transmission_type]).flatten()  # 1D
ownerNo = np.array([ownerNo])  # scalar to 1D
Manufacturer1 = manufacturer_encoder.transform([Manufacturer]).flatten()  # 1D
model1 = model_encoder.transform([Model])  # already 1D usually
mileage = np.array([mileage])  # scalar to 1D
max_power = np.array([max_power])  # scalar to 1D
seats = np.array([seats])  # scalar to 1D
insurance1 = insurance_encoder.transform([insurance_validity]).flatten()  # 1D
engine_cc = np.array([engine_cc])  # scalar to 1D
car_age = np.array([car_age])  # scalar to 1D
city_encoded = ohe_encoder.transform([[city]]).flatten()  # 1D
input_data = np.concatenate([
    fuel1,
    bodytype1,
    kms_driven,
    transmission1,
    ownerNo,
    Manufacturer1,
    model1,
    mileage,
    max_power,
    seats,
    insurance1,
    engine_cc,
    car_age,
    city_encoded
])

# Create dataframe
final_input = pd.DataFrame([input_data])  # Wrap in list for 2D DataFrame

# Predict
if st.button("Predict Price"):
    try:
        prediction = model.predict(final_input)
        st.success(f"💰 Estimated Car Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
