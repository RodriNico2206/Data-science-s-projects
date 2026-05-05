import streamlit as st, pandas as pd, joblib, numpy as np, os
from datetime import datetime
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Ventas - Superstore",
    page_icon="📊",
    layout="wide"
)

# Definir rutas absolutas (cambia esta ruta según tu proyecto)
# Obtener la ruta del directorio actual del script
try:
    # Si el script está en el directorio del proyecto
    SCRIPT_DIR = Path(__file__).parent.absolute()
except NameError:
    # Si se ejecuta en interactive mode (Jupyter, etc.)
    SCRIPT_DIR = Path.cwd()

# O puedes definir la ruta manualmente (descomenta y modifica según tu caso)
# SCRIPT_DIR = Path(r'C:\Users\Usuario\Documents\Data-science-s-projects\superstore-sales-forecasting-thesis')

# Construir rutas absolutas
RUTA_MODELO = SCRIPT_DIR / 'modelo_ventas.pkl'
RUTA_FEATURES = SCRIPT_DIR / 'features.pkl'
RUTA_SCALER = SCRIPT_DIR / 'scaler.pkl'

# Función para verificar y cargar archivos
@st.cache_resource
def cargar_modelo_y_features():
    """Carga el modelo, features y scaler usando rutas absolutas"""
    
    # Verificar que existan los archivos necesarios
    if not RUTA_MODELO.exists():
        st.error(f"❌ No se encuentra el archivo: {RUTA_MODELO}")
        st.info(f"Directorio actual: {SCRIPT_DIR}")
        st.info("Asegúrate de que los archivos .pkl estén en el mismo directorio que este script")
        return None, None, None, False
    
    if not RUTA_FEATURES.exists():
        st.error(f"❌ No se encuentra el archivo: {RUTA_FEATURES}")
        return None, None, None, False
    
    try:
        # Cargar modelo
        modelo = joblib.load(RUTA_MODELO)
        st.success(f"✅ Modelo cargado: {type(modelo).__name__}")
        
        # Cargar features
        features = joblib.load(RUTA_FEATURES)
        st.success(f"✅ Features cargadas: {len(features)} variables")
        
        # Cargar scaler si existe
        if RUTA_SCALER.exists():
            scaler = joblib.load(RUTA_SCALER)
            usar_scaler = True
            st.success("✅ Scaler cargado")
        else:
            scaler = None
            usar_scaler = False
            st.info("⚠️ No se encontró scaler, se usará sin estandarización")
        
        return modelo, features, scaler, usar_scaler
    
    except Exception as e:
        st.error(f"❌ Error al cargar los archivos: {str(e)}")
        return None, None, None, False

# Título de la app
st.title("🎯 Predictor de Ventas - Superstore")
st.markdown("Ingresá los datos de la transacción para estimar las ventas")

# Cargar modelos
modelo, features, scaler, usar_scaler = cargar_modelo_y_features()

# Si no se cargaron los modelos, detener la ejecución
if modelo is None or features is None:
    with st.sidebar:
        st.error("⚠️ No se pudieron cargar los archivos necesarios")
        st.write("**Solución:**")
        st.write("1. Verifica que los archivos existan en:")
        st.code(str(SCRIPT_DIR))
        st.write("2. Archivos necesarios:")
        st.write(f"   - modelo_ventas.pkl: {RUTA_MODELO.exists()}")
        st.write(f"   - features.pkl: {RUTA_FEATURES.exists()}")
    st.stop()

# Definir mapeos para crear variables dummy
categorias = ['Furniture', 'Office Supplies', 'Technology']
subcategorias = ['Bookcases', 'Chairs', 'Copiers', 'Machines', 'Phones', 'Tables']
regiones = ['Central', 'East', 'South', 'West']
segmentos = ['Consumer', 'Corporate', 'Home Office']

# Crear inputs para el usuario
col1, col2 = st.columns(2)

with col1:
    quantity = st.number_input("Cantidad", min_value=1, max_value=50, value=2)
    discount = st.slider("Descuento (%)", 0, 100, 0) / 100
    month = st.selectbox("Mes", range(1, 13))
    day = st.number_input("Día", min_value=1, max_value=31, value=15)
    year = st.number_input("Año", min_value=2020, max_value=2024, value=2024)

with col2:
    category = st.selectbox("Categoría", categorias)
    subcategory = st.selectbox("Subcategoría", subcategorias)
    region = st.selectbox("Región", regiones)
    segment = st.selectbox("Segmento", segmentos)

# Botón para predecir
if st.button("Predecir Ventas", type="primary"):
    try:
        # Crear diccionario con todos los valores
        input_dict = {
            'Quantity': quantity,
            'Discount': discount,
            'Month': month,
            'Day': day,
            'Year': year,
            'DayOfWeek': datetime(year, month, day).weekday(),  # 0=Lunes, 6=Domingo
            'IsWeekend': 1 if datetime(year, month, day).weekday() >= 5 else 0
        }
        
        # Agregar variables dummy (one-hot encoding)
        # Categorías
        for cat in categorias:
            input_dict[f'Category_{cat}'] = 1 if category == cat else 0
        
        # Subcategorías
        for sub in subcategorias:
            input_dict[f'Subcategory_{sub}'] = 1 if subcategory == sub else 0
        
        # Regiones
        for reg in regiones:
            input_dict[f'Region_{reg}'] = 1 if region == reg else 0
        
        # Segmentos
        for seg in segmentos:
            input_dict[f'Segment_{seg}'] = 1 if segment == seg else 0
        
        # Crear DataFrame
        input_data = pd.DataFrame([input_dict])
        
        # Asegurar que todas las features necesarias estén presentes
        for feature in features:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Reordenar columnas para que coincidan con el entrenamiento
        input_data = input_data[features]
        
        # Aplicar scaler si es necesario
        if usar_scaler and scaler is not None:
            # Identificar columnas numéricas
            numeric_cols = ['Quantity', 'Discount', 'Day', 'Month', 'Year', 'DayOfWeek', 'IsWeekend']
            numeric_cols_present = [col for col in numeric_cols if col in input_data.columns]
            if numeric_cols_present:
                input_data[numeric_cols_present] = scaler.transform(input_data[numeric_cols_present])
        
        # Realizar predicción
        prediccion = modelo.predict(input_data)[0]
        
        # Mostrar resultado
        st.success(f"💵 Venta estimada: **${prediccion:,.2f}**")
        
        # Mostrar recomendación basada en el resultado
        st.subheader("📊 Recomendación:")
        if prediccion > 1000:
            st.info("💰 Producto de ALTO valor: Asegurar stock prioritario y considerar envío prioritario")
        elif prediccion > 500:
            st.info("📊 Producto de MEDIO-ALTO valor: Mantener stock regular y monitorear demanda")
        elif prediccion > 200:
            st.info("📦 Producto de MEDIO valor: Stock estándar, ideal para promociones")
        else:
            st.info("🎯 Producto de BAJO valor: Ideal para venta cruzada y paquetes promocionales")
        
        # Mostrar detalles de la predicción
        with st.expander("Ver detalles de la predicción"):
            st.write("**Variables ingresadas:**")
            st.dataframe(input_data)
            st.write(f"**Modelo utilizado:** {type(modelo).__name__}")
            st.write(f"**Features utilizadas:** {len(features)} variables")
            st.write(f"**Ruta del modelo:** {RUTA_MODELO}")
    
    except Exception as e:
        st.error(f"❌ Error al hacer la predicción: {str(e)}")
        st.write("**Posibles causas:**")
        st.write("- Las features del modelo no coinciden con las ingresadas")
        st.write("- El scaler no está configurado correctamente")
        st.write("- Error en los datos de entrada")

# Información del modelo en la barra lateral
with st.sidebar:
    st.header("📈 Información del Modelo")
    st.write(f"**Características:** {len(features)}")
    st.write(f"**Tipo de modelo:** {type(modelo).__name__}")
    st.write(f"**Usa Scaler:** {'Sí' if usar_scaler else 'No'}")
    
    st.subheader("📁 Ubicación de archivos")
    st.write(f"**Directorio:** {SCRIPT_DIR}")
    st.write(f"**Modelo:** {RUTA_MODELO.name}")
    st.write(f"**Features:** {RUTA_FEATURES.name}")
    
    st.subheader("🔝 Variables principales")
    for f in features[:5]:
        st.write(f"- {f}")
    if len(features) > 5:
        st.write(f"... y {len(features)-5} más")
    
    st.header("💡 Consejos")
    st.write("- Descuentos >20% pueden reducir margen")
    st.write("- Tecnología suele tener mayor precio")
    st.write("- Finde semana = mayor actividad comercial")
    
    # Información de depuración (opcional)
    with st.expander("🔧 Diagnóstico"):
        st.write("**Archivos encontrados:**")
        st.write(f"modelo_ventas.pkl: {RUTA_MODELO.exists()}")
        st.write(f"features.pkl: {RUTA_FEATURES.exists()}")
        st.write(f"scaler.pkl: {RUTA_SCALER.exists()}")