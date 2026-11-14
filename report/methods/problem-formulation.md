This study evaluates the ability of off-the-shelf language models to identify wildlife adverts relevant to a specified monitoring policy. Formally, this represents a binary classification task in which a pretrained model maps permitted advert text to a relevance prediction under a policy prompt:

<div class="display-equation" role="region" aria-label="Binary classification mapping" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="f sub theta p maps the text space X to zero or one; predicted y i equals f sub theta p of x i."><mrow><msub><mi>f</mi><mrow><mi>θ</mi><mo>,</mo><mi>p</mi></mrow></msub><mo>:</mo><mi mathvariant="script">X</mi><mo>→</mo><mo>{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>}</mo><mo>,</mo><mspace width="2em"/><msub><mover accent="true"><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><msub><mi>f</mi><mrow><mi>θ</mi><mo>,</mo><mi>p</mi></mrow></msub><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>)</mo><mo>,</mo></mrow></math>
</div>

where:

- <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mi>i</mi></msub><mo>∈</mo><mi mathvariant="script">X</mi></math> is the text supplied for advert <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>i</mi></math>: its title, or its title and description, according to the experimental condition.
- <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>p</mi></math> specifies the monitoring policy prompt. Under few-shot conditions this includes labelled examples.
- <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>θ</mi></math> denotes the pretrained model parameters, which remain unchanged.
- <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>y</mi><mi>i</mi></msub><mo>=</mo><mn>1</mn></math> denotes a policy-relevant advert and <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>y</mi><mi>i</mi></msub><mo>=</mo><mn>0</mn></math> an irrelevant advert; <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mover accent="true"><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub></math> is the model’s prediction.

For models returning a relevance score <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mrow><mi>θ</mi><mo>,</mo><mi>p</mi></mrow></msub><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>)</mo><mo>∈</mo><mo>[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>]</mo></math>, predictions use a fixed decision threshold of 0.5:

<div class="display-equation" role="region" aria-label="Fixed decision threshold" tabindex="0">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Predicted y i equals one if the relevance score is greater than or equal to 0.5, and zero otherwise."><mrow><msub><mover accent="true"><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mn mathvariant="bold">1</mn><mo>[</mo><msub><mi>s</mi><mrow><mi>θ</mi><mo>,</mo><mi>p</mi></mrow></msub><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>)</mo><mo>≥</mo><mn>0.5</mn><mo>]</mo><mo>.</mo></mrow></math>
</div>

The chat baseline instead returns a binary label directly.
