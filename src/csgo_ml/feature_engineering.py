"""
Feature Engineering for CS:GO Dataset
Creates new meaningful features for regression and classification tasks
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_clean_data():
    """Load the cleaned dataset"""
    try:
        # Try to load the cleaned data
        df = pd.read_csv(r"C:\Users\LuisSalamanca\Desktop\Duoc\Machine\csgo-ml\data\02_intermediate\csgo_data_clean.csv")
        print(f"✅ Datos cargados: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Archivo no encontrado. Usando datos raw y aplicando limpieza básica...")
        # Fallback to raw data with basic cleaning
        df_raw = pd.read_csv(r"C:\Users\LuisSalamanca\Desktop\Duoc\Machine\csgo-ml\data\01_raw\Anexo_ET_demo_round_traces_2022.csv", sep=';')
        
        # Basic cleaning
        df = df_raw.copy()
        cols_to_drop = ['TimeAlive', 'TravelledDistance', 'FirstKillTime', 'AbnormalMatch']
        cols_dropped = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_dropped)
        
        # Clean RoundWinner and MatchWinner
        df['RoundWinner'] = df['RoundWinner'].replace('False4', 'False')
        df['RoundWinner'] = df['RoundWinner'].map({'True': True, 'False': False})
        df = df.dropna(subset=['MatchWinner'])
        df['MatchWinner'] = df['MatchWinner'].map({'True': True, 'False': False})
        
        print(f"✅ Datos procesados: {df.shape}")
        return df

def create_performance_features(df):
    """Create performance-based features"""
    print("\n🎯 Creando Features de Rendimiento...")
    
    df_new = df.copy()
    
    # 1. Efficiency Ratios
    df_new['Kill_Equipment_Efficiency'] = df_new['RoundKills'] / (df_new['RoundStartingEquipmentValue'] + 1)
    df_new['Assist_Equipment_Efficiency'] = df_new['RoundAssists'] / (df_new['RoundStartingEquipmentValue'] + 1)
    
    # 2. Performance Ratios
    df_new['Headshot_Kill_Ratio'] = df_new['RoundHeadshots'] / (df_new['RoundKills'] + 1)
    df_new['Assist_Kill_Ratio'] = df_new['RoundAssists'] / (df_new['RoundKills'] + 1)
    df_new['Flank_Kill_Ratio'] = df_new['RoundFlankKills'] / (df_new['RoundKills'] + 1)
    
    # 3. Match-Level Performance
    df_new['Match_Kill_Per_Round'] = df_new['MatchKills'] / df_new['RoundId']
    df_new['Match_Assist_Per_Round'] = df_new['MatchAssists'] / df_new['RoundId']
    df_new['Match_Headshot_Percentage'] = df_new['MatchHeadshots'] / (df_new['MatchKills'] + 1)
    
    # 4. Total Combat Actions
    df_new['Total_Combat_Actions'] = df_new['RoundKills'] + df_new['RoundAssists'] + df_new['RoundHeadshots']
    df_new['Total_Match_Actions'] = df_new['MatchKills'] + df_new['MatchAssists'] + df_new['MatchHeadshots']
    
    # 5. Performance Categories (Binning)
    df_new['Kill_Performance_Level'] = pd.cut(df_new['RoundKills'], 
                                             bins=[-1, 0, 1, 2, 5], 
                                             labels=['No_Kills', 'Single_Kill', 'Double_Kill', 'Multi_Kill'])
    
    df_new['Equipment_Investment_Level'] = pd.cut(df_new['RoundStartingEquipmentValue'],
                                                 bins=[0, 800, 2500, 4500, 10000],
                                                 labels=['Eco', 'Force_Buy', 'Semi_Buy', 'Full_Buy'])
    
    print(f"   ✅ {11} features de rendimiento creadas")
    return df_new

def create_tactical_features(df):
    """Create tactical and weapon-based features"""
    print("\n🔫 Creando Features Tácticas...")
    
    df_new = df.copy()
    
    # 1. Weapon Specialization
    weapon_cols = [col for col in df.columns if col.startswith('Primary')]
    if weapon_cols:
        # Convert to numeric if they're strings
        for col in weapon_cols:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)
        
        # Dominant weapon type
        df_new['Dominant_Weapon'] = df_new[weapon_cols].idxmax(axis=1)
        df_new['Weapon_Specialization_Score'] = df_new[weapon_cols].max(axis=1)
        
        # Weapon diversity (how many weapon types used)
        df_new['Weapon_Diversity'] = (df_new[weapon_cols] > 0).sum(axis=1)
        
        # Specific weapon preferences
        df_new['Prefers_Rifles'] = (df_new['PrimaryAssaultRifle'] > 0.5).astype(int)
        df_new['Prefers_AWP'] = (df_new['PrimarySniperRifle'] > 0.5).astype(int)
        df_new['Prefers_SMG'] = (df_new['PrimarySMG'] > 0.5).astype(int)
    
    # 2. Grenade Usage Patterns
    df_new['Total_Grenades_Used'] = df_new['RLethalGrenadesThrown'] + df_new['RNonLethalGrenadesThrown']
    df_new['Grenade_Lethality_Ratio'] = df_new['RLethalGrenadesThrown'] / (df_new['Total_Grenades_Used'] + 1)
    df_new['Utility_Usage_Ratio'] = df_new['RNonLethalGrenadesThrown'] / (df_new['Total_Grenades_Used'] + 1)
    
    # 3. Grenade Categories
    df_new['Grenade_Usage_Style'] = 'Conservative'
    df_new.loc[df_new['Total_Grenades_Used'] >= 3, 'Grenade_Usage_Style'] = 'Aggressive'
    df_new.loc[(df_new['Total_Grenades_Used'] >= 1) & (df_new['Total_Grenades_Used'] < 3), 'Grenade_Usage_Style'] = 'Moderate'
    
    # 4. Tactical Effectiveness
    df_new['Grenades_Per_Kill'] = df_new['Total_Grenades_Used'] / (df_new['RoundKills'] + 1)
    df_new['Equipment_To_Grenade_Ratio'] = df_new['RoundStartingEquipmentValue'] / (df_new['Total_Grenades_Used'] * 300 + 1)
    
    print(f"   ✅ {12} features tácticas creadas")
    return df_new

def create_temporal_features(df):
    """Create temporal and round progression features"""
    print("\n⏰ Creando Features Temporales...")
    
    df_new = df.copy()
    
    # 1. Round Phases
    df_new['Round_Phase'] = 'Mid'
    df_new.loc[df_new['RoundId'] <= 6, 'Round_Phase'] = 'Early'
    df_new.loc[df_new['RoundId'] >= 25, 'Round_Phase'] = 'Late'
    df_new.loc[df_new['RoundId'] >= 31, 'Round_Phase'] = 'Overtime'
    
    # 2. Economic Rounds (common CS:GO patterns)
    df_new['Is_Pistol_Round'] = ((df_new['RoundId'] == 1) | (df_new['RoundId'] == 16)).astype(int)
    df_new['Is_Anti_Eco_Round'] = ((df_new['RoundId'] == 2) | (df_new['RoundId'] == 17)).astype(int)
    df_new['Is_Buy_Round'] = (df_new['RoundStartingEquipmentValue'] > 3000).astype(int)
    
    # 3. Match Progression
    df_new['Match_Progress_Percentage'] = (df_new['RoundId'] / 30) * 100  # 30 rounds = regulation time
    df_new['Rounds_Remaining'] = np.maximum(0, 30 - df_new['RoundId'])
    
    # 4. Critical Round Indicators
    df_new['Is_Match_Point'] = (df_new['RoundId'] >= 30).astype(int)
    df_new['Is_Critical_Round'] = ((df_new['RoundId'] % 5 == 0) & (df_new['RoundId'] > 10)).astype(int)
    
    # 5. Round Type Classification
    df_new['Round_Type'] = 'Regular'
    df_new.loc[df_new['Is_Pistol_Round'] == 1, 'Round_Type'] = 'Pistol'
    df_new.loc[df_new['Is_Anti_Eco_Round'] == 1, 'Round_Type'] = 'Anti_Eco'
    df_new.loc[df_new['RoundStartingEquipmentValue'] < 1000, 'Round_Type'] = 'Eco'
    df_new.loc[df_new['Is_Match_Point'] == 1, 'Round_Type'] = 'Match_Point'
    
    # 6. Temporal Momentum (performance trends)
    df_new['Early_Round_Performance'] = (df_new['RoundId'] <= 6).astype(int) * df_new['RoundKills']
    df_new['Late_Round_Performance'] = (df_new['RoundId'] >= 25).astype(int) * df_new['RoundKills']
    
    print(f"   ✅ {12} features temporales creadas")
    return df_new

def create_team_economic_features(df):
    """Create team-based and economic features"""
    print("\n💰 Creando Features de Equipo y Económicas...")
    
    df_new = df.copy()
    
    # 1. Economic Features
    df_new['Individual_Equipment_Share'] = df_new['RoundStartingEquipmentValue'] / (df_new['TeamStartingEquipmentValue'] + 1)
    df_new['Team_Economic_Strength'] = pd.cut(df_new['TeamStartingEquipmentValue'],
                                             bins=[0, 5000, 15000, 25000, 40000],
                                             labels=['Very_Poor', 'Poor', 'Average', 'Rich'])
    
    # 2. Equipment Distribution
    df_new['Above_Team_Average_Equipment'] = (df_new['RoundStartingEquipmentValue'] > 
                                             df_new['TeamStartingEquipmentValue'] / 5).astype(int)
    
    df_new['Equipment_Advantage_Ratio'] = df_new['RoundStartingEquipmentValue'] / (df_new['TeamStartingEquipmentValue'] / 5 + 1)
    
    # 3. Team Role Indicators (based on equipment patterns)
    df_new['Likely_Entry_Fragger'] = ((df_new['Individual_Equipment_Share'] > 0.25) & 
                                      (df_new['RoundKills'] > 0)).astype(int)
    
    df_new['Likely_Support_Player'] = ((df_new['RNonLethalGrenadesThrown'] >= 2) & 
                                      (df_new['RoundAssists'] > df_new['RoundKills'])).astype(int)
    
    df_new['Likely_AWPer'] = ((df_new['PrimarySniperRifle'] > 0.7) & 
                             (df_new['RoundStartingEquipmentValue'] > 4000)).astype(int)
    
    # 4. Economic Efficiency
    df_new['Equipment_ROI'] = (df_new['RoundKills'] * 300 + df_new['RoundAssists'] * 150) / (df_new['RoundStartingEquipmentValue'] + 1)
    df_new['Team_Equipment_Efficiency'] = df_new.groupby(['MatchId', 'RoundId', 'Team'])['Equipment_ROI'].transform('mean')
    
    # 5. Relative Performance within Team
    df_new['Relative_Kills_In_Team'] = df_new.groupby(['MatchId', 'RoundId', 'Team'])['RoundKills'].transform(
        lambda x: (x - x.mean()) / (x.std() + 0.01))
    
    df_new['Relative_Equipment_In_Team'] = df_new.groupby(['MatchId', 'RoundId', 'Team'])['RoundStartingEquipmentValue'].transform(
        lambda x: (x - x.mean()) / (x.std() + 0.01))
    
    # 6. Team Coordination Indicators
    df_new['Team_Total_Kills'] = df_new.groupby(['MatchId', 'RoundId', 'Team'])['RoundKills'].transform('sum')
    df_new['Team_Total_Assists'] = df_new.groupby(['MatchId', 'RoundId', 'Team'])['RoundAssists'].transform('sum')
    df_new['Team_Coordination_Score'] = df_new['Team_Total_Assists'] / (df_new['Team_Total_Kills'] + 1)
    
    print(f"   ✅ {14} features de equipo y económicas creadas")
    return df_new

def create_map_specific_features(df):
    """Create map-specific features"""
    print("\n🗺️ Creando Features Específicas de Mapa...")
    
    df_new = df.copy()
    
    # 1. Map-based performance adjustments
    map_stats = df_new.groupby('Map').agg({
        'RoundKills': 'mean',
        'Survived': lambda x: (x == True).mean() if 'Survived' in df_new.columns else 0.5,
        'RoundStartingEquipmentValue': 'mean'
    }).round(3)
    
    df_new = df_new.merge(map_stats, on='Map', suffixes=('', '_map_avg'))
    
    # 2. Performance relative to map average
    df_new['Kills_Above_Map_Average'] = df_new['RoundKills'] - df_new['RoundKills_map_avg']
    df_new['Equipment_Above_Map_Average'] = df_new['RoundStartingEquipmentValue'] - df_new['RoundStartingEquipmentValue_map_avg']
    
    # 3. Map difficulty indicators
    map_difficulty = {
        'de_dust2': 'Balanced',
        'de_inferno': 'CT_Favored', 
        'de_mirage': 'Balanced',
        'de_nuke': 'CT_Favored'
    }
    
    df_new['Map_Balance'] = df_new['Map'].map(map_difficulty)
    
    # 4. Side-specific map features
    df_new['Team_Map_Advantage'] = 'Neutral'
    
    # CT advantages on certain maps
    ct_favored_maps = ['de_inferno', 'de_nuke']
    df_new.loc[(df_new['Map'].isin(ct_favored_maps)) & (df_new['Team'] == 'CounterTerrorist'), 'Team_Map_Advantage'] = 'Favored'
    df_new.loc[(df_new['Map'].isin(ct_favored_maps)) & (df_new['Team'] == 'Terrorist'), 'Team_Map_Advantage'] = 'Disadvantaged'
    
    print(f"   ✅ {7} features específicas de mapa creadas")
    return df_new

def create_interaction_features(df):
    """Create interaction features between different variables"""
    print("\n🔗 Creando Features de Interacción...")
    
    df_new = df.copy()
    
    # 1. Equipment × Performance interactions
    df_new['Equipment_Kills_Interaction'] = df_new['RoundStartingEquipmentValue'] * df_new['RoundKills']
    df_new['Equipment_Survival_Interaction'] = df_new['RoundStartingEquipmentValue'] * df_new['Survived'].astype(int)
    
    # 2. Round × Performance interactions
    df_new['Round_Kills_Interaction'] = df_new['RoundId'] * df_new['RoundKills']
    df_new['Round_Equipment_Interaction'] = df_new['RoundId'] * df_new['RoundStartingEquipmentValue']
    
    # 3. Team × Map interactions
    team_map_combo = df_new['Team'] + '_' + df_new['Map']
    df_new['Team_Map_Combination'] = team_map_combo
    
    # 4. Weapon × Map interactions
    if 'Dominant_Weapon' in df_new.columns:
        df_new['Weapon_Map_Combination'] = df_new['Dominant_Weapon'] + '_' + df_new['Map']
    
    # 5. Economic × Tactical interactions
    df_new['Economic_Tactical_Score'] = df_new['Equipment_Investment_Level'].astype(str) + '_' + df_new['Grenade_Usage_Style']
    
    # 6. Performance synergy features
    df_new['Kills_Assists_Synergy'] = df_new['RoundKills'] * df_new['RoundAssists']
    df_new['Equipment_Grenades_Synergy'] = df_new['RoundStartingEquipmentValue'] * df_new['Total_Grenades_Used']
    
    print(f"   ✅ {9} features de interacción creadas")
    return df_new

def create_statistical_features(df):
    """Create statistical aggregation features"""
    print("\n📊 Creando Features Estadísticas...")
    
    df_new = df.copy()
    
    # 1. Rolling statistics by player (using MatchId + InternalTeamId as proxy for player)
    df_new['Player_ID'] = df_new['MatchId'].astype(str) + '_' + df_new['InternalTeamId'].astype(str)
    
    # Moving averages (using cumulative approach)
    df_new = df_new.sort_values(['Player_ID', 'RoundId'])
    
    df_new['Cumulative_Kills'] = df_new.groupby('Player_ID')['RoundKills'].cumsum()
    df_new['Cumulative_Deaths'] = df_new.groupby('Player_ID')['Survived'].apply(lambda x: (~x.astype(bool)).cumsum())
    df_new['Running_KD_Ratio'] = df_new['Cumulative_Kills'] / (df_new['Cumulative_Deaths'] + 1)
    
    # 2. Recent performance (last 3 rounds)
    for window in [3, 5]:
        df_new[f'Kills_Last_{window}_Rounds'] = df_new.groupby('Player_ID')['RoundKills'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum())
        
        df_new[f'Avg_Equipment_Last_{window}_Rounds'] = df_new.groupby('Player_ID')['RoundStartingEquipmentValue'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
    
    # 3. Match-level aggregations
    df_new['Match_Round_Count'] = df_new.groupby('MatchId')['RoundId'].transform('max')
    df_new['Team_Rounds_Won'] = df_new.groupby(['MatchId', 'Team'])['RoundWinner'].transform('sum')
    df_new['Team_Win_Rate'] = df_new['Team_Rounds_Won'] / df_new['RoundId']
    
    # 4. Performance volatility
    df_new['Kill_Variance'] = df_new.groupby('Player_ID')['RoundKills'].transform(
        lambda x: x.expanding().var().fillna(0))
    
    df_new['Equipment_Variance'] = df_new.groupby('Player_ID')['RoundStartingEquipmentValue'].transform(
        lambda x: x.expanding().var().fillna(0))
    
    print(f"   ✅ {11} features estadísticas creadas")
    return df_new

def main():
    """Main feature engineering pipeline"""
    print("🚀 INICIANDO FEATURE ENGINEERING PARA CS:GO DATASET")
    print("=" * 60)
    
    # Load data
    df = load_clean_data()
    original_shape = df.shape
    
    # Apply all feature engineering steps
    print(f"\n📊 Dataset original: {original_shape}")
    
    df = create_performance_features(df)
    df = create_tactical_features(df)
    df = create_temporal_features(df)
    df = create_team_economic_features(df)
    df = create_map_specific_features(df)
    df = create_interaction_features(df)
    df = create_statistical_features(df)
    
    final_shape = df.shape
    new_features = final_shape[1] - original_shape[1]
    
    print(f"\n🎉 FEATURE ENGINEERING COMPLETADO")
    print("=" * 40)
    print(f"📊 Dataset original: {original_shape}")
    print(f"📊 Dataset final: {final_shape}")
    print(f"✨ Nuevas features creadas: {new_features}")
    print(f"📈 Incremento de columnas: {(new_features/original_shape[1]*100):.1f}%")
    
    # Save enhanced dataset
    output_path = r"C:\Users\LuisSalamanca\Desktop\Duoc\Machine\csgo-ml\data\04_feature\csgo_data_with_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset con features guardado en: {output_path}")
    
    # Create feature summary
    feature_summary = {
        'Performance Features': [
            'Kill_Equipment_Efficiency', 'Assist_Equipment_Efficiency', 'Headshot_Kill_Ratio',
            'Match_Kill_Per_Round', 'Total_Combat_Actions', 'Kill_Performance_Level'
        ],
        'Tactical Features': [
            'Dominant_Weapon', 'Weapon_Specialization_Score', 'Total_Grenades_Used',
            'Grenade_Usage_Style', 'Prefers_Rifles', 'Equipment_To_Grenade_Ratio'
        ],
        'Temporal Features': [
            'Round_Phase', 'Is_Pistol_Round', 'Match_Progress_Percentage',
            'Round_Type', 'Is_Critical_Round', 'Rounds_Remaining'
        ],
        'Economic Features': [
            'Individual_Equipment_Share', 'Team_Economic_Strength', 'Equipment_ROI',
            'Likely_Entry_Fragger', 'Team_Equipment_Efficiency', 'Relative_Kills_In_Team'
        ],
        'Map Features': [
            'Kills_Above_Map_Average', 'Map_Balance', 'Team_Map_Advantage',
            'Equipment_Above_Map_Average'
        ],
        'Interaction Features': [
            'Equipment_Kills_Interaction', 'Team_Map_Combination', 'Kills_Assists_Synergy',
            'Economic_Tactical_Score'
        ],
        'Statistical Features': [
            'Running_KD_Ratio', 'Kills_Last_3_Rounds', 'Team_Win_Rate',
            'Kill_Variance', 'Match_Round_Count'
        ]
    }
    
    print(f"\n📋 RESUMEN DE FEATURES CREADAS:")
    print("=" * 35)
    for category, features in feature_summary.items():
        print(f"\n🔹 {category} ({len(features)} features):")
        for feature in features[:3]:  # Show first 3 of each category
            print(f"   • {feature}")
        if len(features) > 3:
            print(f"   ... y {len(features)-3} más")
    
    return df

if __name__ == "__main__":
    df_enhanced = main()