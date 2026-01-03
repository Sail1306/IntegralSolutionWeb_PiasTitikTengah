import streamlit as st
import numpy as np

try:
    icon = Image.open("assets/icon.png")
except:
    icon = "📈"

st.set_page_config(
    page_title="Integral Solution(Pias Titik Tengah)",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- custom styles ---
st.markdown("""
    <style>
    body { overflow-x: hidden !important; }
    .stButton>button { width: 100%; background-color: #4CAF50; color: white; height: 3em; font-weight: bold; border-radius: 5px; }
    .stTextInput>div>div>input { color: #4CAF50; font-weight: bold; }
    h1, h2, h3 { color: #1565C0; text-align: center; }
    .highlight { background-color: #e8f5e9; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 5px solid #4CAF50; color: #2E7D32; font-size: 1.1em; font-weight: 500; }
    .result-box { background-color: #263238; color: #ECEFF1; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #FF9800; margin: 1rem 0; font-size: 1.1em; font-weight: bold; }
    .function-guide { background-color: #1a1a1a; color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #4CAF50; margin: 1rem 0; }
    .angular-guide { background-color: #e8eaf6; padding: 1rem; border-radius: 4px; border-left: 4px solid #3f51b5; margin: 0.5rem 0; width: 100%; box-sizing: border-box; }
    .angular-button { width: 100%; margin: 0.2rem 0; }
    .code-text { font-family: monospace; background-color: #333333; padding: 0.2rem 0.4rem; border-radius: 3px; color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Helper: Midpoint rule
# --------------------------
def midpoint_rule(f, a, b, n):
    """Composite midpoint rule. f expects scalar input and returns scalar."""
    h = (b - a) / n
    total = 0.0
    for i in range(n):
        x_mid = a + h * (i + 0.5)
        y = f(x_mid)
        total += y
    return total * h

# --------------------------
# Symbolic integration helpers (from your original code)
# --------------------------
def try_integration(expr, x):
    try:
        result = sp.integrate(expr, x, meijerg=True, risch=True)
        if result.has(sp.Integral):
            expr_str = str(expr)
            if 'sin(x**2)' in expr_str:
                S, C = sp.fresnels(x), sp.fresnelc(x)
                result = sp.sqrt(sp.pi/2) * (S + C)
            elif 'exp(-x**2)' in expr_str:
                result = sp.sqrt(sp.pi) * sp.erf(x) / 2
            else:
                simplified = sp.trigsimp(sp.apart(expr))
                result = sp.integrate(simplified, x)
                if result.has(sp.Integral):
                    raise ValueError("Direct integration failed")
        return result
    except:
        try:
            if isinstance(expr, sp.Basic):
                expr_str = str(expr)
                if 'sin(x**2)' in expr_str:
                    return sp.sqrt(sp.pi/2) * (sp.fresnels(x) + sp.fresnelc(x))
                elif 'exp(-x**2)' in expr_str:
                    return sp.sqrt(sp.pi) * sp.erf(x) / 2
            result = sp.integrate(expr, x, manual=True)
            if result.has(sp.Integral):
                raise ValueError("Manual integration failed")
            return result
        except:
            return None

def evaluate_special_function(expr_str, x_val):
    try:
        if 'sin(x**2)' in expr_str:
            s, c = special.fresnel(x_val)
            return np.sqrt(np.pi/2) * (s + c)
        elif 'exp(-x**2)' in expr_str:
            return np.sqrt(np.pi) * special.erf(x_val) / 2
        return None
    except:
        return None

def format_angle(value):
    # Format value in terms of pi when appropriate
    if abs(value) < 1e-10:
        return "0"
    frac = value / np.pi
    if abs(frac - round(frac)) < 1e-10:
        if abs(frac) == 1:
            return "π" if frac > 0 else "-π"
        return f"{int(frac)}π"
    rational = sp.Rational(str(frac)).limit_denominator(12)
    num, den = rational.numerator, rational.denominator
    if abs(num) == 1:
        return f"π/{den}" if num > 0 else f"-π/{den}"
    return f"{num}π/{den}"

# --------------------------
# Plotting helper (uses plotly)
# --------------------------
def create_plot(x_vals, y_vals, expr_str, lower_limit, upper_limit):
    try:
        y_vals = np.asarray(y_vals, dtype=np.float64)
        fig = make_subplots()
        mask = np.isfinite(y_vals)
        fig.add_trace(go.Scatter(
            x=x_vals[mask],
            y=y_vals[mask],
            name="f(x)",
            line=dict(color='#2962FF', width=1.5),
            mode='lines',
            hovertemplate="<b>x</b>: %{x:.4f}<br><b>f(x)</b>: %{y:.4f}",
            hoverlabel=dict(bgcolor="#1565C0", font=dict(size=12, color='white'))
        ))
        fill_mask = (x_vals >= lower_limit) & (x_vals <= upper_limit) & np.isfinite(y_vals)
        if np.any(fill_mask):
            fig.add_trace(go.Scatter(
                x=x_vals[fill_mask],
                y=y_vals[fill_mask],
                fill='tozeroy',
                name="Integration Area",
                line=dict(color='#00C853'),
                fillcolor='rgba(0, 200, 83, 0.2)',
                hoverinfo='skip'
            ))
        fig.add_vline(x=lower_limit, line_width=2, line_dash="dash", line_color="red", annotation_text="Lower Limit", annotation_position="top left")
        fig.add_vline(x=upper_limit, line_width=2, line_dash="dash", line_color="red", annotation_text="Upper Limit", annotation_position="top right")
        vertical_line = {
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": (lower_limit + upper_limit) / 2,
            "y0": 0,
            "x1": (lower_limit + upper_limit) / 2,
            "y1": 1,
            "line": {"color": "blue", "width": 2, "dash": "dot"},
        }
        fig.update_layout(
            shapes=[vertical_line],
            title=dict(text=f"Integration of f(x) = {expr_str}", font=dict(size=14, color='#1565C0'), x=0.5, xanchor='center'),
            xaxis_title="x",
            yaxis_title="f(x)",
            hovermode='closest',
            dragmode='pan',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)'),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', showline=True, linewidth=1, linecolor='rgba(128,128,128,0.8)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', showline=True, linewidth=1, linecolor='rgba(128,128,128,0.8)')
        )
        fig.update_traces(marker=dict(size=12), selector=dict(mode='markers+text'))
        return fig
    except Exception:
        st.error("Error creating plot: The function might be undefined in some regions")
        return None

# --------------------------
# MAIN APP
# --------------------------
def main():
    st.title('Kalkulator Integral')
    st.title('Metode Pias Titik Tengah')

    st.markdown("""
    <div class='highlight'>
    **Welcome to the Integration Calculator!** This tool computes **definite and indefinite** integrals easily.  
    **Made by Saila** 🏆
    </div>
    """, unsafe_allow_html=True)

    # Inputs area
    input_col, guide_col = st.columns([2, 1])
    with input_col:
        st.markdown("### 📝 Enter Your Function")
        expr_str = st.text_input('Function f(x):', value='x**2', help="Use Python/SymPy syntax (e.g., x**2 for x²)")

        limit_type = st.radio("Limit Input Type:", ["Decimal", "Angular (π)"], horizontal=True)

        limit_col1, limit_col2 = st.columns(2)
        with limit_col1:
            if limit_type == "Decimal":
                lower_limit = st.number_input('Lower Limit:', value=0.0, step=0.1, format="%.4f")
            else:
                lower_limit_str = st.text_input('Lower Limit:', value="0", help="Enter as multiples of π (e.g., pi/2, -pi)")
                try:
                    expr_val = lower_limit_str.replace('π', 'pi')
                    lower_limit = float(sp.sympify(expr_val).evalf())
                except:
                    st.error("Invalid input. Examples: pi/2, -pi, 1, 0.5")
                    lower_limit = 0.0

        with limit_col2:
            if limit_type == "Decimal":
                upper_limit = st.number_input('Upper Limit:', value=1.0, step=0.1, format="%.4f")
            else:
                upper_limit_str = st.text_input('Upper Limit:', value="pi/2", help="Enter as multiples of π (e.g., pi/2, -pi)")
                try:
                    expr_val = upper_limit_str.replace('π', 'pi')
                    upper_limit = float(sp.sympify(expr_val).evalf())
                except:
                    st.error("Invalid input. Examples: pi/2, -pi, 1, 0.5")
                    upper_limit = np.pi / 2

        # <-- NEW: jumlah pembagian n (placed with other inputs)
        n = st.number_input(
            "Jumlah Pembagian (n):",
            min_value=1,
            value=100,
            step=1,
            help="Semakin besar n, semakin akurat (tapi lebih lama komputasinya)."
        )

        # quick angular buttons (only affect upper_limit as convenience)
        if limit_type == "Angular (π)":
            st.markdown("<div class='angular-guide'>Common Angular Values:</div>", unsafe_allow_html=True)
            button_cols = st.columns(4)
            angular_values = [("π/6", np.pi/6), ("π/4", np.pi/4), ("π/3", np.pi/3), ("π/2", np.pi/2)]
            for i, (label, val) in enumerate(angular_values):
                with button_cols[i]:
                    if st.button(label, key=f"ang_{i}", use_container_width=True):
                        upper_limit = val

    with guide_col:
        st.markdown("""
        <div class='function-guide' style='padding: 1rem; margin-bottom: 0.5rem;'>
        <h3 style='margin-bottom: 0.5rem; font-size: 1.1em;'>💡 Quick Examples:</h3>
        <ul style='list-style-type: none; padding-left: 0; margin-bottom: 0;'>
            <li style='margin-bottom: 5px; font-size: 0.8em;'>📊 Basic <span class='code-text'>x**2</span></li>
            <li style='margin-bottom: 5px; font-size: 0.8em;'>📐 Trigonometric <span class='code-text'>sin(x)</span></li>
            <li style='margin-bottom: 5px; font-size: 0.8em;'>📈 Exponential <span class='code-text'>exp(-x)</span></li>
            <li style='margin-bottom: 5px; font-size: 0.8em;'>🔄 Complex <span class='code-text'>sin(x**2)</span></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📚 Function Guide", expanded=False):
            st.markdown("""
            <div class='function-guide'>
            <h3>🔢 Basic-Operations</h3>
            • Addition: <span class='code-text'>+</span> (x + 1)<br>
            • Multiplication: <span class='code-text'>*</span> (2*x)<br>
            • Power: <span class='code-text'>**</span> (x**2)<br>
            • Division: <span class='code-text'>/</span> (x/2)<br>
            <h3>🎯 Advanced Functions</h3>
            • Trig: <span class='code-text'>sin(x)</span>, <span class='code-text'>cos(x)</span><br>
            • Exp/Log: <span class='code-text'>exp(x)</span>, <span class='code-text'>log(x)</span><br>
            <h3>🎲 Special</h3>
            • Fresnel: <span class='code-text'>sin(x**2)</span><br>
            • Error func: <span class='code-text'>exp(-x**2)</span><br>
            <h3>Constants</h3> • π: <span class='code-text'>pi</span> • e: <span class='code-text'>E</span>
            </div>
            """, unsafe_allow_html=True)

    # Calculate button (full-width center)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        calculate_button = st.button('🔢 Calculate Integral', type='primary')

    if calculate_button:
        # Basic validation
        if lower_limit >= upper_limit:
            st.error("⚠️ Upper limit must be greater than lower limit")
            return

        # parse expression
        x_sym = sp.symbols('x')
        try:
            expr = sp.sympify(expr_str)
        except Exception:
            st.error("⚠️ Invalid function syntax. Use valid SymPy/Python expression (e.g., x**2, sin(x)).")
            return

        # create numeric function using lambdify (works with numpy)
        try:
            f_num = sp.lambdify(x_sym, expr, modules=['numpy', {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'pi': np.pi, 'erf': special.erf, 'fresnel': special.fresnel, 'fresnels': special.fresnel
            }])
        except Exception:
            st.error("⚠️ Gagal membuat fungsi numerik. Periksa ekspresi.")
            return

        # create x values for plot
        plot_margin = (upper_limit - lower_limit) * 0.2
        x_vals = np.linspace(lower_limit - plot_margin, upper_limit + plot_margin, 1000)
        try:
            y_vals = f_num(x_vals)
            if isinstance(y_vals, tuple):
                y_vals = y_vals[0]
            y_vals = np.asarray(y_vals, dtype=np.float64)
            if np.any(~np.isfinite(y_vals)):
                st.error("⚠️ Fungsi menghasilkan nilai tak hingga atau NaN dalam rentang plotting.")
                return
        except Exception:
            st.error("⚠️ Error menghitung nilai fungsi untuk plotting. Periksa sintaks fungsi.")
            return

        # Attempt symbolic indefinite integral and display
        indefinite_result = try_integration(expr, x_sym)
        if indefinite_result is not None:
            latex_integral = sp.latex(indefinite_result)
            st.markdown(f"### Indefinite Integral:\n$$ \\int {sp.latex(expr)} \\,dx = {latex_integral} + C $$")
        else:
            st.warning("⚠️ Tidak ditemukan integral simbolik yang sederhana (function mungkin terlalu kompleks).")

        # --- NUMERIC: Midpoint Rule ---
        try:
            # ensure f_num works for scalar input (it usually does)
            def f_scalar(t):
                val = f_num(t)
                # if lambdified returns array for scalar, extract
                if isinstance(val, (list, tuple, np.ndarray)):
                    return float(np.array(val).astype(float).item())
                return float(val)
            integral_result = midpoint_rule(f_scalar, lower_limit, upper_limit, int(n))
        except Exception as e:
            st.error("⚠️ Error saat menghitung integral numerik (midpoint). Periksa fungsi atau kurangi n.")
            return

        # create and show plot
        fig = create_plot(x_vals, y_vals, expr_str, lower_limit, upper_limit)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        # Display results
        limit_display_lower = format_angle(lower_limit) if limit_type == "Angular (π)" else f"{lower_limit:.4f}"
        limit_display_upper = format_angle(upper_limit) if limit_type == "Angular (π)" else f"{upper_limit:.4f}"

        st.markdown(f"""
        <div class='result-box'>
        Integration Results:

        - 📊 Function: {expr_str}
        - 📍 Limits: [{limit_display_lower}, {limit_display_upper}]
        - 🔢 Pembagian n: {int(n)}
        - ✨ Definite Integral (Midpoint rule): `{integral_result:.6f}`
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
