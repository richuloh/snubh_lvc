from taipy import Gui
from math import cos,exp
import joblib
import pandas as pd
import lightgbm

model = joblib.load('./models/model_20240212_234310/FOLD_0.pkl')

# df = pd.Series({"AL":26.751644,
#                   "ACD":2.982100,
#                   "LT": 5.233488,
#                   "CCT": 0.577926,
#                   "W2W": 12.131451,
#                   "K1":41.314837,
#                   "K2":44.333060,
#                   "mean_K":42.823949,
#                   "mean_PK":49.625010,
#                   "mean_TK":42.707280,
#                   "TK1": 41.256723,
#                   "TK2": 44.157836,
#                   "PK1": 47.348727,
#                   "PK2": 51.901294
#                   })
# df = pd.DataFrame(df).transpose()

#preds = model.predict(df)[0]


df =pd.DataFrame(pd.Series({"AL":5,
                  "ACD":5,
                  "LT": 5,
                  "CCT": 0.577926,
                  "W2W": 12.131451,
                  "K1":41.314837,
                  "K2":44.333060,
                  "mean_K":42.823949,
                  "mean_PK":49.625010,
                  "mean_TK":42.707280,
                  "TK1": 41.256723,
                  "TK2": 44.157836,
                  "PK1": 47.348727,
                  "PK2": 51.901294
                  })).transpose()

def predict_(model,df_path):
    df = pd.read_csv(df_path)
    print("successful")
    preds = model.predict(df)
    print(preds)
    return preds

preds=0
content = ""
df_path=""
LVC ="Myopic LVC or Not Myopic LVC"


text=""
AL=25
ACD=""
LT=""
W2W=""
CCT=""
K1=""
K2=""
TK1=""
TK2=""
PK1=""
PK2=""

#
Section_1="""
<|{"web_app/logo.png"}|image|width=30vw|>
# Machine learning model for identification of Eyes with history of myopic LVC (laser vision correction) 
### Developed by Richul Oh, M.D. (Seoul National University Hospital)

"""

Section_2="""
<br/>

<|{text}|label=Patient ID|input|><|{AL}|label=AL|number|> <br/>
<|{ACD}|label=ACD|number|><|{LT}|label=LT|number|> <br/>
<|{CCT}|label=CCT|number|><|{W2W}|label=W2W|input|> <br/>
<|{K1}|label=K1|input|><|{K2}|label=K2|input|> <br/>
<|{TK1}|label=TK1|input|><|{TK2}|label=TK2|input|> <br/>
<|{PK1}|label=PK1|input|><|{PK2}|label=PK2|input|> <br/>

<br/>
<|Predict LVC eye|button|on_action=button_pressed|>

<br/>
This eye is <|{LVC}|> eye
<br/>
"""

Section_3="""
<br/>
Select an your CSV file from your file system. 
Make sure that your file is in CSV format

<br/>
<|{content}|file_selector|extensions=.csv|>


# 

<|{preds}|>


"""
#<|label goes here|indicator|value=0|min=0|max=100|width=25vw|>

#csv 로 extension 설정

def on_change(state,var_name,var_val):
    if var_name=="AL":
        state.AL = var_val
        print(AL)
        print(var_val)
    
    
    if var_name =="content":
        state.df_path=var_val
        preds = predict_(model, var_val)[5]
        state.preds = preds
    

def button_pressed(state):
    state.df = pd.DataFrame(pd.Series({"AL":AL,
                  "ACD":5,
                  "LT": 5,
                  "CCT": 0.577926,
                  "W2W": 12.131451,
                  "K1":41.314837,
                  "K2":44.333060,
                  "mean_K":42.823949,
                  "mean_PK":49.625010,
                  "mean_TK":42.707280,
                  "TK1": 41.256723,
                  "TK2": 44.157836,
                  "PK1": 47.348727,
                  "PK2": 51.901294
                  })).transpose()
    
    preds = model.predict(df)[0]
    state.preds=preds
#{} 안에 변수들어감

app = Gui(page=Section_1+Section_2+Section_3)

# value=10
# page="""
# #This is ~~~@@@@
# A value : <|{decay}|>, 
# A slider: <br/>
# <|{decay}|slider|>
# <|{text}|input|> <br/>
# <|{text}|input|> <br/>
# <|{text}|input|> <br/>
# <|{text}|input|> <br/>
# <|{text}|input|> <br/>
# <|{text}|input|> <br/>

# <|{content}|file_selector|>
# """

    
# Gui(page=page).run()

if __name__=="__main__":
    app.run(use_reloader= True)