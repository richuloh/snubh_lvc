import streamlit as st
import pandas as pd
import time
import joblib
import pandas as pd
import lightgbm


model = joblib.load('./models/model_20240212_234310/FOLD_0.pkl')
templete = pd.read_csv("web_app/templete.csv")

st.image('web_app/logo_white.png')
st.title("ML-LVC, the LVC Identifier")
st.header("Machine learning model for identification of Eyes with history of myopic LVC (laser vision correction)")
st.subheader("Developed and copyrighted by Richul Oh, M.D.")
st.markdown("""---""")

    
# st.download_button(
#     label="CSV 로 다운로드"
    
# )


                   
cols = st.columns(2)
data_list=[]

               
                    
keys = ["Pat_ID","ACD", "K1", 'TK1','PK1',"CCT",
        "AL", "LT", "K2","TK2","PK2", "W2W"]
key_textlist = ["Pat_ID","ACD, mm", "K1, D", 'TK1, D','PK1, D',"CCT, mm",
        "AL, mm", "LT,mm", "K2, D","TK2, D","PK2, D", "W2W, mm"]

lists = []
for i, c in enumerate(cols):
        with c:
            for h in range(6):
                key = keys[i*6 + h]
                key_text = key_textlist[i*6 + h]
                a=st.text_input(key_text,key=key)
                # if (i==0) & (h==0):
                    
                # else:
                #     a = st.number_input(key, key=key)
                lists.append(a)
                                    
df = pd.DataFrame(lists)
df = df.transpose()

df.columns = keys


button=st.button("Go for one")
if button:
    st.write(df) 
    
    df = df.drop(["Pat_ID"],axis=1)
    df = df.apply(pd.to_numeric)
        
    df["mean_K"] = (df["K1"]+df["K2"])/2
    df["mean_TK"] = (df["TK1"]+df["TK2"])/2
    df["mean_PK"] = (df["PK1"]+df["PK2"])/2
    preds = model.predict()[0]
    st.write(preds)
    
    if preds > 0.4:
        st.write("This eye might Be a Myopic LVC eye")
    else: 
        st.write("This eye might NOT be a Myopic LVC eye")
    

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
        
button_all=st.button("Go for ALL")
if button_all:
    df = df.drop(["Pat_ID"],axis=1)
    df = df.apply(pd.to_numeric)
        
    df["mean_K"] = (df["K1"]+df["K2"])/2
    df["mean_TK"] = (df["TK1"]+df["TK2"])/2
    df["mean_PK"] = (df["PK1"]+df["PK2"])/2
    
    st.write("Prediction takes a few second or a few minute. Please wait")
    preds = model.predict(df)
    df["LVC"] = pd.Series(preds).apply(lambda x: "Myopic LVC" if x>0.4 else "Not Myopic LVC")
    st.write("Prediction Completed")
    

st.download_button(
    label = "Download the Results",
    data= df.to_csv().encode('utf-8'),
    file_name = 'results.csv',
    mime="text")
   
