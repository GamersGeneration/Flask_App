var form = document.querySelector('form');
form.addEventListener('submit', function(event) {
  event.preventDefault();
  var formData = new FormData(form);
  fetch('/predict', {
    method: 'POST',
    body: formData
  }).then(function(response) {
    return response.json();
  }).then(function(data) {
    var prediction = data['prediction'];
    var predictionDiv = document.querySelector('#prediction');
    predictionDiv.textContent = 'Prediction: ' + prediction;
  });
});
