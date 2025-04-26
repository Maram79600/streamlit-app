
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# تحميل النموذج المدرب
rf_model = joblib.load('random_forest_model.pkl')  # تأكد من حفظ النموذج أولاً

# عنوان التطبيق
st.title("نموذج تصنيف الأمراض")

# إضافة مدخلات للمستخدم
st.sidebar.header("الرجاء إدخال البيانات:")

age = st.sidebar.slider("العمر", 18, 100)
sex = st.sidebar.selectbox("الجنس", ["ذكر", "أنثى"])
cholesterol = st.sidebar.number_input("مستوى الكوليسترول", min_value=0)
blood_pressure = st.sidebar.number_input("ضغط الدم", min_value=0)

# تحويل القيم المدخلة إلى بيانات قابلة للاستخدام
inputs = {
    'age': age,
    'sex': 1 if sex == 'ذكر' else 0,  # تحويل الجنس إلى قيمة عددية
    'cholesterol': cholesterol,
    'blood_pressure': blood_pressure
}

input_df = pd.DataFrame([inputs])

# تطبيع البيانات (نفس الخطوات التي تم استخدامها في تدريب النموذج)
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_df)  # تطبيع البيانات المدخلة

# التنبؤ باستخدام النموذج المدرب
prediction = rf_model.predict(input_scaled)

# عرض النتيجة للمستخدم
if prediction == 1:
    st.write("التنبؤ: الشخص مصاب بالمرض.")
else:
    st.write("التنبؤ: الشخص غير مصاب بالمرض.")
