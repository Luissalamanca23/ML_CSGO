# Business Understanding - Problema de Regresión

## Contexto del Proyecto
**Cliente:** Valve Corporation  
**Tipo de Proyecto:** Análisis de Datos CS:GO - Problema de Regresión  
**Dataset:** 79,157 registros de 333 partidas CS:GO en 4 mapas  

## Objetivos de Negocio - Regresión

### Objetivos Principales
1. **Predicción de Rendimiento Individual**: Predecir métricas continuas de rendimiento de jugadores
2. **Optimización de Inversión en Equipamiento**: Predecir valores óptimos de equipamiento para maximizar resultados
3. **Análisis de Eficiencia**: Predecir métricas de efectividad basadas en patrones de juego

### Preguntas de Negocio Específicas
1. **¿Cuántas eliminaciones realizará un jugador en una ronda?** (RoundKills: 0-5)
2. **¿Cuántas asistencias tendrá un jugador en una ronda?** (RoundAssists: 0-4)  
3. **¿Cuál es el valor óptimo de equipamiento inicial?** (RoundStartingEquipmentValue: 0-8850)
4. **¿Cuántas eliminaciones conseguirá en toda la partida?** (MatchKills: 0-41)
5. **¿Cuántos headshots realizará por partida?** (MatchHeadshots: 0-22)

### Variables Objetivo (Continuas) Identificadas
- **RoundKills** (0-5): Eliminaciones por ronda - Media: 0.7
- **RoundAssists** (0-4): Asistencias por ronda - Media: 0.1  
- **RoundHeadshots** (0-5): Headshots por ronda - Media: 0.3
- **MatchKills** (0-41): Eliminaciones totales - Media: 8.5
- **MatchHeadshots** (0-22): Headshots totales - Media: 3.9
- **RoundStartingEquipmentValue** (0-8850): Inversión en equipamiento - Media: 3778
- **TeamStartingEquipmentValue** (0-36150): Inversión del equipo - Media: 18890

### Variables Predictoras Disponibles
**Categóricas:**
- Map (4 mapas: de_inferno, de_dust2, de_mirage, de_nuke)
- Team (Terrorist/Counter-Terrorist)
- RoundId (1-36: progreso de la partida)

**Numéricas:**
- RLethalGrenadesThrown (0-4): Granadas letales
- RNonLethalGrenadesThrown (0-6): Granadas no letales
- PrimaryAssaultRifle, PrimarySniperRifle, PrimaryHeavy, PrimarySMG (proporción de uso)

### Casos de Uso de Negocio
1. **Estrategia de Equipamiento**: Predecir inversión óptima para maximizar kills/assists
2. **Evaluación de Jugadores**: Predecir rendimiento esperado basado en patrones históricos
3. **Planificación Táctica**: Estimar resultados de diferentes configuraciones de equipo
4. **Análisis de ROI**: Relacionar inversión en equipamiento con rendimiento

### Criterios de Éxito
- **R² > 0.80** para predicciones de RoundKills y MatchKills
- **MAE < 1.0** para predicciones de kills por ronda
- **R² > 0.70** para predicciones de equipamiento óptimo
- **Interpretabilidad**: Modelos que permitan identificar factores clave

### Restricciones y Consideraciones
- **Datos Eliminados**: TimeAlive, TravelledDistance, FirstKillTime (formato europeo corrupto)
- **Balance de Datos**: Dataset balanceado entre Terrorist/Counter-Terrorist
- **Mapas**: Distribución desigual (de_inferno: 36%, otros ~24% cada uno)
- **Estacionalidad**: Considerar progreso de ronda (early/mid/late game)

### Entregables Esperados
1. Modelos de regresión para predicción de kills/assists
2. Sistema de recomendación de inversión en equipamiento
3. Análisis de factores que maximizan rendimiento individual
4. Dashboard de métricas predictivas para entrenadores