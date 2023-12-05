import pandas as pd

file_path = '../../Users/82109/Downloads/2019_kbo_for_kaggle_v2.csv'
df = pd.read_csv(file_path)


# Print the top 10 players in hits (안타, H), batting average (타율, avg), homerun (홈런, HR), and on-
# base percentage (출루율, OBP) for each year from 2015 to 2018.
def print_top_players(df, year):
    targets = ['H', 'avg', 'HR', 'OBP']

    target_year = df[df['year'] == year]

    print(f'Year: {year}')
    for target in targets:
        results = target_year.nlargest(10, target)[['batter_name', target]]
        print(results)


# Print the player with the highest war (승리  기여도) by position (cp) in 2018.
def print_top_war_players(df, year):
    targets = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

    target_year = df[df['year'] == year]

    print(f'Year: {year}')
    for targets in targets:
        selected_target = target_year[target_year['cp'] == targets]
        result = selected_target.nlargest(1, 'war')[['batter_name', 'cp', 'war']]
        print(result)
        print()


# Among R (득점), H (안타), HR (홈런), RBI (타점), SB (도루), war (승리  기여도), avg (타율), OBP
# (출루율), and SLG (장타율), which has the highest correlation with salary (연봉)?
def find_highest_correlation(df):
    targets = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']

    corr = df[targets].corrwith(df['salary'])

    result_target = corr.idxmax()
    result_value = corr[result_target]

    return result_target, result_value


if __name__ == '__main__':
    print('##########start 2-1-1##########')
    print(
        'Print the top 10 players in hits (안타, H), batting average (타율, avg), homerun (홈런, HR), and on- base percentage (출루율, OBP) for each year from 2015 to 2018.')
    print()
    for year in range(2015, 2019):
        print_top_players(df, year)
    print('\n##########end 2-1-1##########\n')

    print('##########start 2-1-2##########')
    print('Print the player with the highest war (승리  기여도) by position (cp) in 2018')
    print()
    print_top_war_players(df, 2018)
    print('\n##########end 2-1-2##########\n')

    print('##########start 2-1-3##########')
    print(
        'Among R (득점), H (안타), HR (홈런), RBI (타점), SB (도루), war (승리  기여도), avg (타율), OBP (출루율), and SLG (장타율), which has the highest correlation with salary (연봉)?')
    print()
    result_target, result_value = find_highest_correlation(df)
    print(f'result_target: {result_target} \nresult_value: {result_value}')
    print('\n##########end 2-1-3##########\n')
