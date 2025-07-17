# Business Understanding - Problema de Clasificación

## Contexto del Proyecto
**Cliente:** Valve Corporation  
**Tipo de Proyecto:** Análisis de Datos CS:GO - Problema de Clasificación  
**Dataset:** 79,157 registros de 333 partidas CS:GO en 4 mapas  

## Objetivos de Negocio - Clasificación

### Objetivos Principales
1. **Predicción de Supervivencia**: Determinar si un jugador sobrevivirá la ronda
2. **Predicción de Resultados**: Determinar ganador de ronda y partida
3. **Análisis de Ventajas**: Identificar factores que determinan victoria por equipo/mapa
4. **Detección de Patrones**: Clasificar situaciones de juego y estrategias exitosas

### Preguntas de Negocio Específicas
1. **¿Sobrevivirá un jugador esta ronda?** (Survived: 40% sí, 60% no)
2. **¿Qué equipo ganará la ronda?** (RoundWinner: balanceado 50/50)
3. **¿Qué equipo ganará la partida?** (MatchWinner: 47% vs 53%)
4. **¿Qué mapa favorece más a cada lado?** (Map-specific performance)
5. **¿En qué rondas es más probable ganar?** (Round progression analysis)

### Variables Objetivo (Categóricas) Identificadas

**Variables Principales:**
- **Survived** (Binaria): False=47,214 | True=31,943 (40% supervivencia)
- **RoundWinner** (Binaria): False≈39,588 | True≈39,568 (balanceado)
- **MatchWinner** (Binaria): False=42,017 | True=37,139 (47% vs 53%)

**Variables Contextuales:**
- **Team** (Binaria): Terrorist≈39,591 | Counter-Terrorist≈39,564 (balanceado)
- **Map** (Multiclase): 
  - de_inferno: 28,869 (36%)
  - de_dust2: 19,120 (24%)
  - de_mirage: 19,019 (24%)  
  - de_nuke: 12,149 (15%)

### Variables Predictoras para Clasificación

**Features Categóricas:**
- Team (Terrorist/Counter-Terrorist)
- Map (4 opciones)
- RoundId (1-36: early/mid/late game)

**Features Numéricas:**
- RoundKills (0-5): Impacto directo en supervivencia
- RoundAssists (0-4): Soporte al equipo
- RoundHeadshots (0-5): Habilidad técnica
- RoundStartingEquipmentValue (0-8850): Ventaja económica
- TeamStartingEquipmentValue (0-36150): Ventaja del equipo
- RLethalGrenadesThrown (0-4): Uso táctico
- RNonLethalGrenadesThrown (0-6): Utilidad

**Features de Armas:**
- PrimaryAssaultRifle: Proporción de uso de rifles
- PrimarySniperRifle: Uso de rifles de francotirador
- PrimaryHeavy: Armas pesadas
- PrimarySMG: Armas submáquina

### Casos de Uso de Negocio

1. **Análisis Táctico en Tiempo Real**:
   - Predecir probabilidad de supervivencia durante la ronda
   - Alertas para situaciones de alto riesgo

2. **Estrategia de Equipo**:
   - Identificar configuraciones de equipo con mayor probabilidad de victoria
   - Optimizar selección de mapa basada en fortalezas del equipo

3. **Análisis de Meta-juego**:
   - Detectar patrones de victoria por mapa
   - Identificar desequilibrios entre Terrorist/Counter-Terrorist

4. **Sistema de Recomendaciones**:
   - Sugerir estrategias basadas en equipamiento disponible
   - Recomendar posicionamiento según probabilidades de supervivencia

### Métricas de Éxito

**Para Survived (Desbalanceado 60/40):**
- **Precision/Recall > 0.75** para clase minoritaria (Survived=True)
- **F1-Score > 0.70** para ambas clases
- **AUC-ROC > 0.80**

**Para RoundWinner/MatchWinner (Balanceado):**
- **Accuracy > 0.70**
- **F1-Score > 0.70** para ambas clases
- **AUC-ROC > 0.75**

**Para Map Classification:**
- **Macro F1-Score > 0.65** (4 clases)
- **Accuracy > 0.60**

### Desafíos y Consideraciones

**Desbalance de Datos:**
- Survived: 40% positivos (requiere técnicas de balanceo)
- Distribución desigual por mapas (de_inferno dominante)

**Datos Problemáticos Identificados:**
- RoundWinner tiene valor inconsistente 'False4' (limpiar)
- MatchWinner tiene NaN values (imputar o eliminar)
- Variables temporal (TimeAlive) inutilizables

**Dependencias Temporales:**
- RoundId indica progreso (early rounds vs late rounds)
- Efectos de momentum entre rondas consecutivas

### Entregables Esperados

1. **Modelo de Supervivencia**: Predictor binario de supervivencia de jugador
2. **Predictor de Resultados**: Clasificador de ganador de ronda/partida  
3. **Análisis de Mapas**: Clasificador y análisis de ventajas por mapa
4. **Dashboard de Probabilidades**: Interface para decisiones tácticas en tiempo real
5. **Reporte de Factores Críticos**: Features más importantes para cada tipo de clasificación