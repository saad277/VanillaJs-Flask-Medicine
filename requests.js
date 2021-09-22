let stage = document.querySelector("#pressureStage");
let medication = document.querySelector("#medication");

function randomForest() {
  if (
    !(
      gender !== null &&
      age !== null &&
      headache !== null &&
      breadth !== null &&
      visual !== null &&
      nose !== null &&
      blood !== null &&
      range !== null &&
      dRange !== null
    )
  ) {
    alert("Select All options");
    return;
  }
  fetch("http://192.168.0.106:5000/rf", {
    headers: {
      "Content-Type": "application/json",
    },
    method: "POST",
    body: JSON.stringify({
      gender,
      age,
      headache,
      breadth,
      visual,
      nose,
      blood,
      range,
      dRange,
    }),
  })
    .then((res) => res.json())
    .then((res) => {
      stage.innerHTML = res.prediction;
      medication.innerHTML = MEDICINES[res.prediction];
    });
}

const MEDICINES = {
  "HYPERTENSION (Stage-1)": "Medicine: Norvac",
  "HYPERTENSION (Stage-2)": "medicine: Sofvasc",
  "HYPERTENSIVE CRISIS": "medicine: Losarten",
  "HYPERTENSION (Stage-1).": "medicine: Lipiget",
  "HYPERTENSION (Stage-2).": "medicine: Natrilix",
  "HYPERTENSIVE CRISIS.": "medicine: Benefol",
};
