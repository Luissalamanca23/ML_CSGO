# Data Understanding - Análisis Exploratorio CS:GO Dataset

## Resumen del Dataset
**Archivo:** `Anexo_ET_demo_round_traces_2022.csv`  
**Dimensiones:** 79,157 filas × 30 columnas  
**Contexto:** Datos extraídos de replays de 333 partidas CS:GO  
**Período:** 2022  

## Descripción General de los Datos

### Estructura del Dataset
- **Granularidad:** Cada fila representa un jugador en una ronda específica
- **Partidas:** 333 partidas únicas (MatchId: 4-511)
- **Rondas:** Hasta 36 rondas por partida (RoundId: 1-36)
- **Jugadores:** 10 jugadores por ronda (5 Terrorist + 5 Counter-Terrorist)
- **Mapas:** 4 mapas oficiales de CS:GO

### Distribución por Mapas
| Mapa | Registros | Porcentaje |
|------|-----------|------------|
| de_inferno | 28,869 | 36.4% |
| de_dust2 | 19,120 | 24.1% |
| de_mirage | 19,019 | 24.0% |
| de_nuke | 12,149 | 15.3% |

### Distribución por Equipos
- **Terrorist:** 39,591 registros (50.0%)
- **Counter-Terrorist:** 39,564 registros (50.0%)
- **Valores nulos:** 2 registros (0.003%)

## Calidad de Datos - Problemas Críticos Identificados

### 🚨 Columnas INUTILIZABLES (Formato Europeo Corrupto)

**TimeAlive, TravelledDistance, FirstKillTime:**
```
Ejemplos de valores:
- TimeAlive: '51.120.248.995.704.500'
- TravelledDistance: '10.083.140.737.457.000'  
- FirstKillTime: '58.006.269.999.999.900'
```
**Problema:** Formato numérico europeo con múltiples puntos como separadores de miles.  
**Estado:** ELIMINAR - No convertibles a valores numéricos útiles.

### 🔧 Columnas que REQUIEREN LIMPIEZA

**RoundWinner:**
- Valores: ['False', 'True', 'False4']
- Problema: Valor inconsistente 'False4' (1 registro)
- Solución: Mapear 'False4' → False

**MatchWinner:**
- Valores: ['True', 'False', NaN]
- Problema: 1 valor nulo
- Solución: Imputar o eliminar registro

**Primary Weapon Columns:**
- Contienen valores decimales extraños (ej: 0.9615384615384616)
- Deben representar proporciones de uso de armas
- Requiere validación y normalización

### ❌ Columnas CONSTANTES (Sin Valor Predictivo)

**AbnormalMatch:**
- Valor único: False (100% de los casos)
- Acción: ELIMINAR

## Análisis de Variables Objetivo

### Variables de Clasificación

**Survived (Supervivencia del Jugador):**
- False: 47,214 (59.6%)
- True: 31,943 (40.4%)
- **Observación:** Dataset desbalanceado hacia no-supervivencia

**RoundWinner (Ganador de Ronda):**
- False: 39,588 (50.0%)
- True: 39,568 (50.0%)
- **Observación:** Perfectamente balanceado

**MatchWinner (Ganador de Partida):**
- False: 42,017 (53.1%)
- True: 37,139 (46.9%)
- **Observación:** Ligero desbalance hacia False

### Variables de Regresión

| Variable | Min | Max | Media | Descripción |
|----------|-----|-----|-------|-------------|
| RoundKills | 0 | 5 | 0.7 | Eliminaciones por ronda |
| RoundAssists | 0 | 4 | 0.1 | Asistencias por ronda |
| RoundHeadshots | 0 | 5 | 0.3 | Headshots por ronda |
| MatchKills | 0 | 41 | 8.5 | Eliminaciones totales |
| MatchHeadshots | 0 | 22 | 3.9 | Headshots totales |
| RoundStartingEquipmentValue | 0 | 8,850 | 3,778 | Inversión individual |
| TeamStartingEquipmentValue | 0 | 36,150 | 18,890 | Inversión del equipo |

## Análisis de Variables Predictoras

### Variables Categóricas
- **Map:** 4 categorías bien distribuidas
- **Team:** 2 categorías balanceadas
- **RoundId:** 36 valores (1-36), representa progreso temporal

### Variables Numéricas de Conteo
- **RLethalGrenadesThrown:** 0-4 (granadas letales)
- **RNonLethalGrenadesThrown:** 0-6 (granadas utilitarias)
- **Round Performance:** Kills, Assists, Headshots, FlankKills

### Variables de Equipamiento
- **Equipment Values:** Bien distribuidos, representan economía del juego
- **Primary Weapons:** Proporciones de uso de diferentes tipos de armas

## Patrones y Observaciones Clave

### 1. Distribución de Rendimiento
```
RoundKills:
- 0 kills: 54.5% (mayoría de jugadores no eliminan)
- 1 kill: 29.0%
- 2+ kills: 16.5% (rendimiento excepcional)
```

### 2. Economía del Juego
```
RoundStartingEquipmentValue más frecuentes:
- $4,700: 14.1% (compra completa)
- $200: 8.3% (eco round)
- $5,500: 6.9% (AWP + utilidades)
```

### 3. Uso de Granadas
```
RNonLethalGrenadesThrown:
- 0: 36.0% (sin utilidades)
- 1-2: 43.6% (uso estándar)
- 3+: 20.4% (uso intensivo)
```

## Recomendaciones para Preparación de Datos

### Acciones Inmediatas
1. **ELIMINAR:** TimeAlive, TravelledDistance, FirstKillTime, AbnormalMatch
2. **LIMPIAR:** RoundWinner (corregir 'False4'), MatchWinner (manejar NaN)
3. **VALIDAR:** Primary weapon columns (verificar rangos 0-1)

### Ingeniería de Features Sugerida
1. **Equipment Efficiency:** RoundKills / RoundStartingEquipmentValue
2. **Round Phase:** Early (1-6), Mid (7-20), Late (21-36)
3. **Team Economic State:** Eco, Force-buy, Full-buy basado en TeamEquipmentValue
4. **Weapon Specialization:** Dominant weapon type per player

### Estrategia de Validación
1. **Temporal Split:** Separar por MatchId para evitar data leakage
2. **Stratified Sampling:** Mantener distribución de mapas y equipos
3. **Cross-Validation:** K-fold considerando estructura jerárquica (Match > Round > Player)

## Limitaciones Identificadas
1. **Información Temporal Perdida:** Sin timestamps precisos utilizables
2. **Contexto de Ronda:** Sin información de objetivo específico (bomb sites)
3. **Skill Level:** Sin datos de ranking o nivel de habilidad de jugadores
4. **Resultado Económico:** Sin seguimiento de dinero post-ronda