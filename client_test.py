import requests

patients = [
    {
        "id": "P001",
        "profile": {
            "age": 70,
            "sex": "M"
        },
        "vitals": {
            "resting_bp": 165,
            "max_hr": 105
        },
        "labs": {
            "cholesterol": 410,
            "st_depression": 2.8
        },
        "clinical": {
            "chest_pain_type": 3
        }
    },
    {
        "id": "P002",
        "profile": {
            "age": 38,
            "sex": "F"
        },
        "vitals": {
            "resting_bp": 110,
            "max_hr": 175
        },
        "labs": {
            "cholesterol": 180,
            "st_depression": 0.2
        },
        "clinical": {
            "chest_pain_type": 0
        }
    },
    {
        "id": "P003",
        "profile": {
            "age": 57,
            "sex": "M"
        },
        "vitals": {
            "resting_bp": 140,
            "max_hr": 135
        },
        "labs": {
            "cholesterol": 260,
            "st_depression": 1.1
        },
        "clinical": {
            "chest_pain_type": 2
        }
    }
]

URL = "http://127.0.0.1:5000/predict"

print("Local Heart Disease API – Batch Test")
print("=" * 60)

for p in patients:
    features = [
        p["profile"]["age"],
        p["vitals"]["resting_bp"],
        p["labs"]["cholesterol"],
        p["vitals"]["max_hr"],
        p["labs"]["st_depression"],
        p["clinical"]["chest_pain_type"]
    ]

    response = requests.post(URL, json={"features": features})

    print(f"\nPatient {p['id']}")
    print(f"  Feature vector: {features}")

    if response.status_code == 200:
        result = response.json()
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Risk: {result['risk']}")
    else:
        print(f"  Error {response.status_code}: {response.text}")

print("\n" + "=" * 60)
print("Batch test completed.")
