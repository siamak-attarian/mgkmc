> From: https://gemini.google.com/app/984ca39812a87e68

# you asked

So for a homogenous elastic system with E=70 GPa and nu=0.3 that is stretched by 8% in the x directions, calculate the sigma_xx and sigma_yy in these three conditions:

1-fully plain strain
2-plain strain in z direction, but sigma_yy=0
3-fully plain stress

---

# gemini response

```python?code_reference&code_event_index=2
import numpy as np

E = 70e9 # Pa
nu = 0.3
eps_xx = 0.08

# Case 1: Fully Plane Strain (epsilon_yy = 0, epsilon_zz = 0)
# sigma_xx = [E / ((1+nu)*(1-2nu))] * (1-nu) * eps_xx
# sigma_yy = [E / ((1+nu)*(1-2nu))] * nu * eps_xx
factor1 = E / ((1 + nu) * (1 - 2 * nu))
sigma_xx_1 = factor1 * (1 - nu) * eps_xx
sigma_yy_1 = factor1 * nu * eps_xx

# Case 2: Plane strain in z (epsilon_zz = 0), but sigma_yy = 0
# epsilon_zz = 0 => sigma_zz = nu * (sigma_xx + sigma_yy) = nu * sigma_xx
# epsilon_xx = (1/E) * (sigma_xx - nu*sigma_zz) = (1/E) * (sigma_xx - nu^2 * sigma_xx) = (sigma_xx / E) * (1 - nu^2)
sigma_xx_2 = (E * eps_xx) / (1 - nu**2)
sigma_yy_2 = 0.0

# Case 3: Fully Plane Stress (sigma_zz = 0)
# Usually implies sigma_yy = 0 for a uniaxial test in x
sigma_xx_3 = E * eps_xx
sigma_yy_3 = 0.0

print(f"Case 1: sigma_xx = {sigma_xx_1/1e9:.2f} GPa, sigma_yy = {sigma_yy_1/1e9:.2f} GPa")
print(f"Case 2: sigma_xx = {sigma_xx_2/1e9:.2f} GPa, sigma_yy = {sigma_yy_2/1e9:.2f} GPa")
print(f"Case 3: sigma_xx = {sigma_xx_3/1e9:.2f} GPa, sigma_yy = {sigma_yy_3/1e9:.2f} GPa")


```
```text?code_stdout&code_event_index=2
Case 1: sigma_xx = 7.54 GPa, sigma_yy = 3.23 GPa
Case 2: sigma_xx = 6.15 GPa, sigma_yy = 0.00 GPa
Case 3: sigma_xx = 5.60 GPa, sigma_yy = 0.00 GPa

```
To calculate the stresses $\sigma_{xx}$ and $\sigma_{yy}$ for the given homogeneous elastic system, we use the generalized Hooke's Law. 

**Given Parameters:**
* Young's Modulus ($E$): $70 \text{ GPa}$
* Poisson's Ratio ($\nu$): $0.3$
* Applied Strain ($\epsilon_{xx}$): $0.08$ (8%)

---

### 1. Fully Plane Strain
In a "fully plane strain" condition (often called uniaxial strain), all lateral strains are constrained to zero.
* **Conditions:** $\epsilon_{yy} = 0$ and $\epsilon_{zz} = 0$.
* **Formula for $\sigma_{xx}$:** $\sigma_{xx} = \frac{E(1-\nu)}{(1+\nu)(1-2\nu)} \epsilon_{xx}$
* **Formula for $\sigma_{yy}$:** $\sigma_{yy} = \frac{E\nu}{(1+\nu)(1-2\nu)} \epsilon_{xx}$

**Calculation:**
$$\sigma_{xx} = \frac{70 \times (1 - 0.3)}{(1 + 0.3)(1 - 2 \times 0.3)} \times 0.08 = \frac{49}{0.52} \times 0.08 \approx 7.538 \text{ GPa}$$
$$\sigma_{yy} = \frac{70 \times 0.3}{0.52} \times 0.08 \approx 3.231 \text{ GPa}$$

**Results:**
* $\sigma_{xx} = 7.54 \text{ GPa}$
* $\sigma_{yy} = 3.23 \text{ GPa}$

---

### 2. Plane Strain in $z$ direction, but $\sigma_{yy} = 0$
This is a mixed condition where the $z$-direction is constrained (plane strain), but the $y$-direction is free to expand or contract.
* **Conditions:** $\epsilon_{zz} = 0$ and $\sigma_{yy} = 0$.
* **Relation:** For $\epsilon_{zz} = 0$, we have $\sigma_{zz} = \nu(\sigma_{xx} + \sigma_{yy})$. Since $\sigma_{yy} = 0$, then $\sigma_{zz} = \nu \sigma_{xx}$.
* **Formula for $\sigma_{xx}$:** Substituting into $\epsilon_{xx} = \frac{1}{E}[\sigma_{xx} - \nu(\sigma_{yy} + \sigma_{zz})]$, we get:
    $$\epsilon_{xx} = \frac{1}{E}[\sigma_{xx} - \nu(0 + \nu \sigma_{xx})] = \frac{\sigma_{xx}}{E}(1 - \nu^2) \implies \sigma_{xx} = \frac{E \epsilon_{xx}}{1 - \nu^2}$$

**Calculation:**
$$\sigma_{xx} = \frac{70 \times 0.08}{1 - 0.3^2} = \frac{5.6}{0.91} \approx 6.154 \text{ GPa}$$

**Results:**
* $\sigma_{xx} = 6.15 \text{ GPa}$
* $\sigma_{yy} = 0 \text{ GPa}$

---

### 3. Fully Plane Stress
In a "fully plane stress" condition (standard uniaxial stress), all lateral stresses are zero. The material is free to deform in both $y$ and $z$ directions.
* **Conditions:** $\sigma_{yy} = 0$ and $\sigma_{zz} = 0$.
* **Formula for $\sigma_{xx}$:** $\sigma_{xx} = E \epsilon_{xx}$

**Calculation:**
$$\sigma_{xx} = 70 \times 0.08 = 5.6 \text{ GPa}$$

**Results:**
* $\sigma_{xx} = 5.60 \text{ GPa}$
* $\sigma_{yy} = 0 \text{ GPa}$

---

### Summary Table

| Condition | $\sigma_{xx}$ (GPa) | $\sigma_{yy}$ (GPa) |
| :--- | :--- | :--- |
| **1. Fully Plane Strain** | 7.54 | 3.23 |
| **2. Plane Strain ($z$), $\sigma_{yy}=0$** | 6.15 | 0.00 |
| **3. Fully Plane Stress** | 5.60 | 0.00 |