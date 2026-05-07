import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import calendar

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Ventas - Superstore",
    page_icon="📊",
    layout="wide"
)

# Definir rutas absolutas
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
except NameError:
    SCRIPT_DIR = Path.cwd()

# Construir rutas absolutas
RUTA_MODELO = SCRIPT_DIR / 'modelo_ventas.pkl'
RUTA_FEATURES = SCRIPT_DIR / 'features.pkl'
RUTA_SCALER = SCRIPT_DIR / 'scaler.pkl'

# Función mejorada para cargar modelo
@st.cache_resource
def cargar_modelo_y_features():
    """Carga el modelo, features y scaler - Maneja tanto modelos directos como diccionarios"""
    
    if not RUTA_MODELO.exists():
        st.error(f"❌ No se encuentra el archivo: {RUTA_MODELO}")
        return None, None, None, False
    
    try:
        contenido = joblib.load(RUTA_MODELO)
        
        if isinstance(contenido, dict):
            st.info("📦 Extrayendo modelo del diccionario...")
            if 'modelo' in contenido:
                modelo = contenido['modelo']
                st.success(f"✅ Modelo extraído: {type(modelo).__name__}")
                
                if not RUTA_FEATURES.exists() and 'caracteristicas_entrenamiento' in contenido:
                    features = contenido['caracteristicas_entrenamiento']
                    st.success(f"✅ Features extraídas: {len(features)}")
                else:
                    features = None
            else:
                st.error("❌ No se encontró 'modelo' en el diccionario")
                return None, None, None, False
        else:
            modelo = contenido
            st.success(f"✅ Modelo cargado: {type(modelo).__name__}")
            features = None
        
        if RUTA_FEATURES.exists():
            features = joblib.load(RUTA_FEATURES)
            st.success(f"✅ Features cargadas: {len(features)} variables")
        elif features is None:
            if hasattr(modelo, 'feature_names_in_'):
                features = list(modelo.feature_names_in_)
                st.success(f"✅ Features del modelo: {len(features)}")
            else:
                st.error("❌ No se encontró features.pkl")
                return None, None, None, False
        
        if RUTA_SCALER.exists():
            scaler = joblib.load(RUTA_SCALER)
            usar_scaler = True
            st.success("✅ Scaler cargado")
        else:
            scaler = None
            usar_scaler = False
            st.info("⚠️ Sin estandarización")
        
        return modelo, features, scaler, usar_scaler
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None, None, False

def validar_dia(mes, año, dia):
    """Valida que el día exista en el mes/año dado"""
    if mes and año and dia:
        _, ultimo_dia = calendar.monthrange(año, mes)
        return min(dia, ultimo_dia)
    return dia

# Título
st.title("🎯 Predictor de Ventas - Superstore")
st.markdown("Ingresá los datos de la transacción para estimar las ventas")

# Cargar modelos
modelo, features, scaler, usar_scaler = cargar_modelo_y_features()

if modelo is None or features is None:
    with st.sidebar:
        st.error("⚠️ No se pudieron cargar los archivos necesarios")
    st.stop()

# Definir mapeos
categorias = ['Furniture', 'Office Supplies', 'Technology']
subcategorias = ['Bookcases', 'Chairs', 'Copiers', 'Machines', 'Phones', 'Tables']
regiones = ['Central', 'East', 'South', 'West']
segmentos = ['Consumer', 'Corporate', 'Home Office']

# Crear inputs - CAMPOS VACÍOS AL INICIO
col1, col2 = st.columns(2)

with col1:
    quantity = st.number_input("📦 Cantidad", value=None, step=1, placeholder="Ej: 2")
    discount = st.slider("💰 Descuento (%)", 0, 100, 0) / 100
    month = st.selectbox("📅 Mes", options=[None] + list(range(1, 13)), 
                        format_func=lambda x: "Seleccione mes..." if x is None else calendar.month_name[x])
    day = st.number_input("📆 Día", value=None, step=1, placeholder="Ej: 15")
    year = st.number_input("📅 Año", value=None, step=1, placeholder="Ej: 2024")

with col2:
    category = st.selectbox("🏷️ Categoría", options=[None] + categorias,
                           format_func=lambda x: "Seleccione categoría..." if x is None else x)
    subcategory = st.selectbox("📌 Subcategoría", options=[None] + subcategorias,
                              format_func=lambda x: "Seleccione subcategoría..." if x is None else x)
    region = st.selectbox("🌍 Región", options=[None] + regiones,
                         format_func=lambda x: "Seleccione región..." if x is None else x)
    segment = st.selectbox("👥 Segmento", options=[None] + segmentos,
                          format_func=lambda x: "Seleccione segmento..." if x is None else x)

# Validación de campos completos
def validar_campos():
    """Verifica que todos los campos necesarios estén completos"""
    campos_faltantes = []
    
    if quantity is None:
        campos_faltantes.append("Cantidad")
    if month is None:
        campos_faltantes.append("Mes")
    if day is None:
        campos_faltantes.append("Día")
    if year is None:
        campos_faltantes.append("Año")
    if category is None:
        campos_faltantes.append("Categoría")
    if subcategory is None:
        campos_faltantes.append("Subcategoría")
    if region is None:
        campos_faltantes.append("Región")
    if segment is None:
        campos_faltantes.append("Segmento")
    
    return campos_faltantes

# Validar día del mes si hay mes y año
if month and year and day:
    dia_valido = validar_dia(month, year, day)
    if day != dia_valido:
        st.warning(f"⚠️ {calendar.month_name[month]} no tiene {day} días. Se ajustará a {dia_valido}")
        day = dia_valido

# Botón para predecir
if st.button("🔮 Predecir Ventas", type="primary", use_container_width=True):
    # Verificar campos completos
    campos_faltantes = validar_campos()
    
    if campos_faltantes:
        st.error(f"❌ Por favor complete los siguientes campos: {', '.join(campos_faltantes)}")
    else:
        try:
            # Crear diccionario con todos los valores
            input_dict = {
                'Quantity': quantity,
                'Discount': discount,
                'Month': month,
                'Day': day,
                'Year': year,
                'DayOfWeek': datetime(year, month, day).weekday(),
                'IsWeekend': 1 if datetime(year, month, day).weekday() >= 5 else 0
            }
            
            # Agregar variables dummy
            for cat in categorias:
                input_dict[f'Category_{cat}'] = 1 if category == cat else 0
            
            for sub in subcategorias:
                input_dict[f'Subcategory_{sub}'] = 1 if subcategory == sub else 0
            
            for reg in regiones:
                input_dict[f'Region_{reg}'] = 1 if region == reg else 0
            
            for seg in segmentos:
                input_dict[f'Segment_{seg}'] = 1 if segment == seg else 0
            
            # Crear DataFrame
            input_data = pd.DataFrame([input_dict])
            
            # Asegurar todas las features
            for feature in features:
                if feature not in input_data.columns:
                    input_data[feature] = 0
            
            input_data = input_data[features]
            
            # Aplicar scaler si es necesario
            if usar_scaler and scaler is not None:
                numeric_cols = ['Quantity', 'Discount', 'Day', 'Month', 'Year', 'DayOfWeek', 'IsWeekend']
                numeric_cols_present = [col for col in numeric_cols if col in input_data.columns]
                if numeric_cols_present:
                    input_data[numeric_cols_present] = scaler.transform(input_data[numeric_cols_present])
            
            # Predicción
            prediccion = modelo.predict(input_data)[0]
            
            # Mostrar resultado
            st.success(f"💵 Venta estimada: **${prediccion:,.2f}**")
            
            # Recomendaciones
            st.subheader("📊 Recomendación:")
            if prediccion > 1000:
                st.info("💰 Producto de ALTO valor: Asegurar stock prioritario")
            elif prediccion > 500:
                st.info("📊 Producto de MEDIO-ALTO valor: Mantener stock regular")
            elif prediccion > 200:
                st.info("📦 Producto de MEDIO valor: Stock estándar")
            else:
                st.info("🎯 Producto de BAJO valor: Ideal para venta cruzada")
            
            # === SOLUCIÓN 1: DETALLES MEJORADOS ===
            with st.expander("📋 Ver detalles de la predicción"):
                st.write("**Variables ingresadas:**")
                
                # Configurar dataframe con ancho máximo y scroll horizontal
                st.dataframe(
                    input_data,
                    use_container_width=True,  # Usa todo el ancho disponible
                    height=300  # Altura fija con scroll vertical
                )
                
                st.write(f"**Modelo utilizado:** {type(modelo).__name__}")
                st.write(f"**Features utilizadas:** {len(features)} variables")
                st.write(f"**Ruta del modelo:** {RUTA_MODELO}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Barra lateral
with st.sidebar:
    st.header("📈 Información del Modelo")
    st.write(f"**Características:** {len(features)}")
    st.write(f"**Tipo:** {type(modelo).__name__}")
    st.write(f"**Scaler:** {'Sí' if usar_scaler else 'No'}")
    
    st.header("💡 Consejos")
    st.write("- Descuentos >20% pueden reducir margen")
    st.write("- Tecnología suele tener mayor precio")
    st.write("- Finde semana = mayor actividad comercial")