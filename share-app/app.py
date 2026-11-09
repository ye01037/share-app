# 의료비 예측 데모 앱 : 입력 → 예측 → 시각화
import joblib
import altair as alt
import pandas as pd
import streamlit as st

COLS = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
SEX = {'남성': 'male', '여성': 'female'}
SMOKER = {'흡연': 'yes', '비흡연': 'no'}
REGION = {'북동부': 'northeast', '북서부': 'northwest', '남동부': 'southeast', '남서부': 'southwest'}
# 학습 데이터(train.csv)에 있던 범위. 이 밖은 모델이 본 적 없는 값
RANGE = {'age': (18, 64), 'bmi': (15.96, 53.13), 'children': (0, 5)}
NAME = {'age': '나이', 'bmi': 'BMI', 'children': '자녀 수'}

st.set_page_config(page_title='의료비 예측', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load('model.joblib')


model = load_model()

# ── 입력 ─────────────────────────────────────────
with st.sidebar:
    st.header('가입자 정보')
    age = st.number_input('나이', min_value=0, max_value=100, value=35)
    sex = st.radio('성별', list(SEX), horizontal=True)
    bmi = st.number_input('BMI', min_value=10.0, max_value=70.0, value=31.0, step=0.5)
    children = st.number_input('자녀 수', min_value=0, max_value=10, value=1)
    smoker = st.radio('흡연 여부', list(SMOKER), horizontal=True)
    region = st.selectbox('지역', list(REGION), index=2)

person = {'age': age, 'sex': SEX[sex], 'bmi': bmi, 'children': children,
          'smoker': SMOKER[smoker], 'region': REGION[region]}


def predict(**change):
    row = pd.DataFrame([{**person, **change}])[COLS]
    return float(model.predict(row)[0])


# ── 예측 ─────────────────────────────────────────
st.title('1년 의료비 예측')
st.caption('보험 가입자 1,070명의 기록으로 학습한 모델이 계산한 추정치입니다. 평균 오차는 약 2,400달러입니다.')

pred = predict()
other = '비흡연' if smoker == '흡연' else '흡연'
diff = predict(smoker=SMOKER[other]) - pred
c1, c2 = st.columns(2)
c1.metric('예상 1년 의료비', f'{pred:,.0f}달러')
c2.metric(f'{other}이라면', f'{pred + diff:,.0f}달러', f'{diff:+,.0f}달러', delta_color='inverse')

out = [f'{NAME[k]} {person[k]}' for k, (lo, hi) in RANGE.items() if not lo <= person[k] <= hi]
if out:
    st.warning(f'학습 데이터 범위를 벗어난 입력이 있습니다 ({", ".join(out)}). '
               '모델이 본 적 없는 값이라 예측을 믿기 어렵습니다.')

# ── 시각화 ───────────────────────────────────────
st.subheader('값 하나만 바꿔 보면')
key = st.radio('바꿔 볼 값', ['BMI', '나이'], horizontal=True)
col = 'bmi' if key == 'BMI' else 'age'
lo, hi = RANGE[col]
xs = [round(lo + (hi - lo) * i / 80, 2) for i in range(81)]
curve = pd.DataFrame({key: xs, '예상 의료비': [predict(**{col: x}) for x in xs]})
now = pd.DataFrame({key: [person[col]], '예상 의료비': [pred]})
line = alt.Chart(curve).mark_line(color='#8843FF', strokeWidth=3).encode(
    x=alt.X(key, scale=alt.Scale(zero=False)), y=alt.Y('예상 의료비', title='예상 의료비 (달러)'))
dot = alt.Chart(now).mark_point(size=220, filled=True, color='#1A1A1A').encode(x=key, y='예상 의료비')
st.altair_chart(line + dot, use_container_width=True)
st.caption(f'다른 정보는 그대로 두고 {key}만 학습 데이터 범위({lo}~{hi}) 안에서 바꿨을 때의 예측입니다. 검은 점이 지금 입력입니다.')
