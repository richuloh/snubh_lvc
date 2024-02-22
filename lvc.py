import streamlit as st
import pandas as pd
import time
import joblib
import pandas as pd
import lightgbm
import numpy as np

model_0 = joblib.load('./models/FOLD_0.pkl')
model_1 = joblib.load('./models/FOLD_1.pkl')
model_2 = joblib.load('./models/FOLD_2.pkl')
model_3 = joblib.load('./models/FOLD_3.pkl')
model_4 = joblib.load('./models/FOLD_4.pkl')

models = [model_0, model_1, model_2, model_3,model_4]
templete = pd.read_csv("web_app/templete.csv")

st.image('web_app/logo_white.png')
st.title("ML-LVC, the LVC Identifier")
st.header("Machine learning model for identification of Eyes with history of myopic LVC (laser vision correction)")
st.subheader("Developed and copyrighted by Richul Oh, M.D.")
st.markdown("""---""")

st.caption("Make sure if there is NO empty input box. If there are unknown biometric value, the predicted value is NOT RELIABLE!")
    
# st.download_button(
#     label="CSV 로 다운로드"
    
# )


                   
cols_1 = st.columns(2)
cols_2 = st.columns(2)
cols_3 = st.columns(2)
cols_4 = st.columns(2)
cols_5 = st.columns(2)
cols_6 = st.columns(2)

col_list = [cols_1,cols_2, cols_3, cols_4,cols_5,cols_6]

data_list=[]

                  
keys = ["Pat_ID","AL", "ACD",'LT', "K1", 'K2', 'TK1','TK2','PK1','PK2','CCT','W2W']
key_textlist = ["Pat_ID","AL, mm", "ACD, mm", "LT,mm", "K1, D", "K2, D", 'TK1, D',"TK2, D",
                'PK1, D',"PK2, D","CCT, mm","W2W, mm"]

lists = []

col_order = ['AL', 'ACD', 'LT', 'CCT','W2W','K1','K2','mean_K','PK1','PK2','mean_PK','mean_TK','TK1','TK2']
for h, cols in enumerate(col_list):
    for i, c in enumerate(cols):
        with c:

            key = keys[i + h*2]
            key_text = key_textlist[i  + h*2]
            a=st.text_input(key_text,key=key)
            # if (i==0) & (h==0):
                
            # else:
            #     a = st.number_input(key, key=key)
            lists.append(a)
                    
# for i, c in enumerate(cols_2):
#         with c:
#             for h in range(1):
#                 key = keys[i + h]
#                 key_text = key_textlist[i + h]
#                 a=st.text_input(key_text,key=key)
#                 # if (i==0) & (h==0):
                    
#                 # else:
#                 #     a = st.number_input(key, key=key)
#                 lists.append(a)

df = pd.DataFrame(lists)
df = df.transpose()

df.columns = keys
st.write(df)


button=st.button("Run for one")
if button:
    st.write(df) 
    
    df = df.drop(["Pat_ID"],axis=1)
    df = df.apply(pd.to_numeric)
        
    df["mean_K"] = (df["K1"]+df["K2"])/2
    df["mean_TK"] = (df["TK1"]+df["TK2"])/2
    df["mean_PK"] = (df["PK1"]+df["PK2"])/2
    
    df = df[col_order]
    
    preds_list =[]
    for model in models:
        preds_list.append(model.predict(df)[0])
        
    final_preds=np.round(np.mean(preds_list),4)
    st.write("According to ML-LVC model, the probability of Myopic LVC eyes is: ", final_preds)
           
    if final_preds > 0.3811:
        st.write("The probability exceeded our cut-off value. This eye might Be a Myopic LVC eye")
    else: 
        st.write("The probability did not exceed our cut-off value. This eye might NOT be a Myopic LVC eye")
    

st.markdown("""---""")
st.write("For research purpose, you can upload your dataset")
st.write("Note that the dataset should follow the templete. You can download the templete here")

st.download_button(
    label = "Download Templete",
    data= templete.to_csv().encode('utf-8'),
    file_name = 'templete.csv',
    mime="text")


# 파일 업로드 버튼 (업로드 기능)
file = st.file_uploader("Select your CSV file", type=['csv']) #, 'xls', 'xlsx'])


# Excel or CSV 확장자를 구분하여 출력하는 경우
if file is not None:
    ext = file.name.split('.')[-1]
    if ext == 'csv':
        # 파일 읽기
        df = pd.read_csv(file)
        # 출력
        st.dataframe(df.head(5))
        st.write("Upload Completed")
    # elif 'xls' in ext:
    #     # 엑셀 로드
    #     df = pd.read_excel(file, engine='openpyxl')
    #     # 출력
    #     st.dataframe(df)
    #     print("Upload Completed")
        
button_all=st.button("Run for ALL")
if button_all:
    df = df.drop(["Pat_ID"],axis=1)
    df = df.apply(pd.to_numeric)
        
    df["mean_K"] = (df["K1"]+df["K2"])/2
    df["mean_TK"] = (df["TK1"]+df["TK2"])/2
    df["mean_PK"] = (df["PK1"]+df["PK2"])/2
    
    df = df[col_order]
    
    st.write("Prediction takes a few second or a few minute. Please wait")
    
    preds_list =[]
    for model in models:
        preds_list.append(model.predict(df))
    
    final_preds=np.array(preds_list).mean(axis=0)

    df["preds"] = final_preds

    df["LVC"] = df["preds"] .apply(lambda x: "Myopic LVC" if x > 0.3811 else "Not Myopic LVC")
    st.write("Prediction Completed")
    

st.download_button(
    label = "Download the Results",
    data= df.to_csv().encode('utf-8'),
    file_name = 'results.csv',
    mime="text")
   
