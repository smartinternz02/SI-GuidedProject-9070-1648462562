from __future__ import division, print_function
import os
import numpy as np
from keras.models import load_model
from keras import utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/home')

l={0:"Oopps!! Your apple plant is infected by Black Rots. This infection is a fungal infection. To control balck rot, remove the cankers by pruning at least 15 inches below the end and burn or bury them. Treating the sites with the antibiotic streptomycin or a copper-based fungicide will be helpful.",
  1:"Yaayy!! Your apple plant is healthy. But, maintain the soil pH of 6.0 to 7.0 for healthy growth. Avoid planting apples in a low spot where cold air or frost can settle.",
2:"Oopps!! Your corn plant is infected by Northern Leaf Blight. The primary management strategy to reduce the incidence and severity of NCLB is planting resistant products. Using fungicides is also helpful.",
3:"Yaayy!! Your corn plant is healthy. But, maintain the soil consistently moist, but not soggy and only need fertilizer every 6 months. It prefers temperatures of 75 to 80 degrees F.",
4:"Oopps!! Your peach plant is infected by Bacterial Spots. This is a difficult disease to control when environmental conditions favor pathogen spread. Compounds for the treatment include copper, oxytetracycline (Mycoshield and generic equivalents), and syllit+captan; however, repeated applications are typically necessary for even minimal disease control.",
5:"Yaayy!! Your peach plant is healthy. But, you should have deep sandy soil that ranges from a loam to a clay loam for healthy growth. Poor drainage in the soil will kill the root system of growing peach trees, so make sure the soil is well drained. Growing peach trees prefer a soil pH of around 6.5."}

l1={0:"Oopps!! Your pepper plant is infected by Bacterial Leaft Spot. The disease cycle can be stopped by using the Sango formula for disinfectants. Bleach treatment and hot water treatment is also helpful.",
    1:"Yaayy!! Your pepper plant is healthy. But, take the necessary precautions like, putting the plant where it gets at least 10 hours of direct sunlight. Keep soil evenly moist for good growth. Peppers need well draining soil that is rich and loamy, but avoid too much nitrogen in the soil. Too much nitrogen can cause plenty of leaves and little to no peppers. Your soil should have a pH between 6.0 and 6.5.",
    2:"Oopps!! Your potato plant is Early Blight. Avoid irrigation in cool cloudy weather and time irrigation to allow plants time to dry before nightfall. Protectant fungicides (e.g. maneb, mancozeb, chlorothalonil, and triphenyl tin hydroxide) are effective.",
    3:"Oopps!! Your potato plant is Late Blight. The late blight can be effectively managed with prophylactic spray of mancozeb, cymoxanil+mancozeb or dimethomorph+mancozeb.",
    4:"Yaayy!! Your potato plant is healthy. But, take the necessary precautions like, putting the plant where it gets at least 10 hours of direct sunlight. Potatoes do best in well-drained and fertile soil. Maintain the pH between 5.0 and 5.5. Keep soil evenly moist for good growth. Do not add large amounts of organic matter to the soil as it may contribute to potato scab, a disease that frequently infects potatoes.",
    5:"Oopps!! Your tomato plant is effected by bacterial spots. To protect the uninfected plants remove the infected leaves and bury or burn them s there is no cure for this infection. To prevent future infections plant pathogen-free seeds or transplants to prevent the introduction of bacterial spot pathogens on contaminated seed or seedlings.",
    6:"Oopps!! Your tomato plant is late blight. Early treatment for this disease is needed. Fungicides like e Daconil fungicides from GardenTech brand prevent, stop, and control late blight and more than 65 types of fungal disease. Planting resistant cultivars and watering the plants early in the mornings help to prevent this infection.",
    7:"Oopps!! Your tomato plant has leaf molds. Watering the plants early in the mornings help them to get sufficient time to dry out. Fungicidal sprays mostly calcium chloride based sparys help in getting rid of leaf molds.",
    8:"Oopps!! Your tomato plant is infected by Septoria leaf spot. Removing the infected leaves immediately will curb the spread of infection. Organic and chemical fungicides with chlorothalonil are effective in treatment."}
def model_predict(img_path,q):
    if(q=="fruits"):
        model=load_model('fruits10.h5')
        img = utils.load_img(img_path, target_size=(128, 128))
        x = utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds=np.argmax(model.predict(x), axis=1)
        return l[preds[0]]
    else:
        model=load_model('Veg.h5')
        img = utils.load_img(img_path, target_size=(128, 128))
        x = utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds=np.argmax(model.predict(x), axis=1)
        return l1[preds[0]]
        

@app.route('/home', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("hi")
    if request.method == 'POST':
        print("SEKHAR")
        # Get the file from post request
        f = request.files['fileq']
        #q=request.files['Choose']
        q=request.form.get("Choose")

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path,q)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(preds)
        print(result)          
        #if(str(result)=='[[1. 0.]]'):
           # result=" NO PNEMONIA, you are safe"
       # else:
          #  result="PNEMONIA,Please consider doctor"
        return render_template('indexr.html',data=result)
    return None


if __name__ == '__main__':
    app.run(debug=True)
    