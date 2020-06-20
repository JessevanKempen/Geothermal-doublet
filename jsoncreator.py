import json

data = {}
data['aquifer'] = []
data['aquifer'].append({
    'd_top': 900,
    'labda': 0.031,
    'H': 200,
    'T_surface': 20,
    'porosity': 0.05,
    'rho_f': 1000,
    'mhu': 8.9e-4,
    'K': 4e-13
})
data['well'] = []
data['well'].append({
    'r': 0.076,
    'Q': 7.5,
    'L': 1000,
    'Ti_inj': 30,
    'epsilon': 0.046,
})

with open('parameters.txt', 'w') as outfile:
    json.dump(data, outfile)