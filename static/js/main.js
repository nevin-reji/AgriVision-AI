document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('crop-form');
  if (!form) return;

  form.addEventListener('submit', (event) => {
    const inputs = Array.from(form.querySelectorAll('input'));
    let hasInvalid = false;

    inputs.forEach((input) => {
      if (!input.value || Number.isNaN(Number(input.value))) {
        hasInvalid = true;
        input.style.borderColor = '#b94141';
      } else {
        input.style.borderColor = '#d9dfd9';
      }
    });

    if (hasInvalid) {
      event.preventDefault();
      alert('Please enter valid numeric values before submitting the form.');
    }
  });
});
