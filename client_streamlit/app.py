import streamlit as st
import requests
import os
# URL de tu API
#API_URL = "http://localhost:8000/predict"
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
# Configuración de la página
st.set_page_config(page_title="Tienda de Entregas - Predicción de Delay", layout="wide")

# Cabecera tipo franja
st.markdown("""
<div style="background-color:#77a9c5; padding: 10px; border-radius: 5px; text-align: center; color: white;">
    <h2 style="margin: 0;">Maestría de IA</h2>
    <h4 style="margin: 0;">Efraín Salinas - Rodrigo Cardenas</h4>
    
</div>
""", unsafe_allow_html=True)

st.title("🚚 Tienda de Entregas - Predicción de Retraso")

st.markdown("""
Bienvenido al cliente de predicción de **Delay** para pedidos.  
Complete los datos del pedido, el cliente y el vendedor para obtener la predicción.
""")

with st.form("delay_form"):
    st.subheader("👤 Cliente")
    col1, col2 = st.columns(2)
    with col1:
        customer_geolocation_lat = st.number_input("Latitud Cliente (customer_geolocation_lat)", value=-23.5)
        customer_geolocation_zip_code_prefix = st.number_input("ZIP Cliente (customer_geolocation_zip_code_prefix)", value=12345)
        order_purchase_timestamp_year = st.number_input("Año Compra (order_purchase_timestamp_year)", min_value=2000, max_value=2100, value=2025)
        order_purchase_timestamp_dayofweek = st.number_input("Día de la Semana (order_purchase_timestamp_dayofweek)", min_value=0, max_value=6, value=0)
        review_score = st.number_input("Puntuación Review (review_score)", min_value=1.0, max_value=5.0, value=5.0)
        order_purchase_timestamp_day = st.number_input("Día Compra (order_purchase_timestamp_day)", min_value=1, max_value=31, value=1)
    with col2:
        customer_geolocation_lng = st.number_input("Longitud Cliente (customer_geolocation_lng)", value=-46.6)
        customer_zip_code_prefix = st.number_input("ZIP Entrega (customer_zip_code_prefix)", value=12345)
        order_purchase_timestamp_month = st.number_input("Mes Compra (order_purchase_timestamp_month)", min_value=1, max_value=12, value=1)
        freight_value = st.number_input("Valor Envío (freight_value)", value=10.0)
        price = st.number_input("Precio Producto (price)", value=50.0)
        product_weight_g = st.number_input("Peso Producto (product_weight_g)", value=500.0)

    st.subheader("🏪 Vendedor")
    col3, col4 = st.columns(2)
    with col3:
        seller_geolocation_lat = st.number_input("Latitud Vendedor (seller_geolocation_lat)", value=-23.6)
        seller_zip_code_prefix = st.number_input("ZIP Code Vendedor (seller_zip_code_prefix)", value=12345)
    with col4:
        seller_geolocation_lng = st.number_input("Longitud Vendedor (seller_geolocation_lng)", value=-46.7)
        seller_geolocation_zip_code_prefix = st.number_input("ZIP Logístico (seller_geolocation_zip_code_prefix)", value=12345)
        payment_value = st.number_input("Valor Pago (payment_value)", value=100.0)

    st.subheader("📦 Producto")
    col5, col6, col7 = st.columns(3)
    with col5:
        product_height_cm = st.number_input("Alto (cm) (product_height_cm)", value=10.0)
    with col6:
        product_width_cm = st.number_input("Ancho (cm) (product_width_cm)", value=10.0)
    with col7:
        product_length_cm = st.number_input("Largo (cm) (product_length_cm)", value=10.0)

    submitted = st.form_submit_button("🔮 Predecir Delay")

    if submitted:
        payload = {
            "customer_geolocation_lat": customer_geolocation_lat,
            "order_purchase_timestamp_month": order_purchase_timestamp_month,
            "freight_value": freight_value,
            "customer_geolocation_lng": customer_geolocation_lng,
            "customer_geolocation_zip_code_prefix": customer_geolocation_zip_code_prefix,
            "customer_zip_code_prefix": customer_zip_code_prefix,
            "review_score": review_score,
            "seller_geolocation_lng": seller_geolocation_lng,
            "seller_zip_code_prefix": seller_zip_code_prefix,
            "seller_geolocation_zip_code_prefix": seller_geolocation_zip_code_prefix,
            "seller_geolocation_lat": seller_geolocation_lat,
            "payment_value": payment_value,
            "order_purchase_timestamp_year": order_purchase_timestamp_year,
            "price": price,
            "product_weight_g": product_weight_g,
            "order_purchase_timestamp_day": order_purchase_timestamp_day,
            "product_height_cm": product_height_cm,
            "product_width_cm": product_width_cm,
            "product_length_cm": product_length_cm,
            "order_purchase_timestamp_dayofweek": order_purchase_timestamp_dayofweek
        }

        # Llamada a la API
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                st.success(f"✅ Predicción: {response.json()}")
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ No se pudo conectar con la API: {e}")
