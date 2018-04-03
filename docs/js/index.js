const model = new KerasJS.Model({
    filepath: './model_lite.bin',
    gpu: false
});

model.ready()
    .then(() => {
        $('#loading').hide();
    })
    .catch(err => {
        console.log(err);

        $('#loading').text('โหลด Model ไม่สำเร็จ');
    });

$(document).ready(initPage);

function initPage() {
    $('#message').keyup(onBtnPredictClicked);
}

function onBtnPredictClicked() {
    var text = $('#message').val();
    var input = gen_input(text);

    var batch = [];
    (async () => {

        var predictions = [];
        for (var i = 0; i < text.length; i++) {
            //https://github.com/transcranial/keras-js/issues/27#issuecomment-260856925
            var input_1 = new Float32Array(input["input_1"][i]);
            var input_2 = new Float32Array(input["input_2"][i]);
            var inputData = { "input_1": input_1, "input_2": input_2 };
            var output = await model.predict(inputData);

            predictions.push(output.dense_2[0]);
        }

        result = parse_prediction(text, predictions);
        console.log("compelted");
        $('#result').html(result.join(" | "));
    })();

}

function gen_input(text) {
    text_pad = '          ' + text + '          ';
    var text_pad = text_pad.split('');
    var n = text.length;
    var n_pad = 21;
    var n_pad_2 = 10;

    var character_list = Array();
    for (var i = n_pad_2; i < n_pad_2 + n; i++) {
        a = text_pad.slice(i + 1, i + n_pad_2 + 1);
        b = text_pad.slice(i - n_pad_2, i).reverse();
        c = text_pad.slice(i, i + 1);
        var char_list = a.concat(b).concat(c);
        character_list[i - n_pad_2] = char_list;
    }


    X_char = Array();
    X_type = Array();
    for (var i = 0; i < character_list.length; i++) {
        characters = character_list[i];
        var x_char = Array();
        var x_type = Array();
        for (var j = 0; j < characters.length; j++) {
            var type = CHAR_TYPE_FLATTEN[characters[j]];
            if (typeof type == 'undefined') {
                type = 'o';
            }

            x_char[j] = CHARS_MAP[characters[j]];
            x_type[j] = CHAR_TYPES_MAP[type];
        }
        X_char[i] = x_char;
        X_type[i] = x_type;
    }

    return { "input_1": X_char, "input_2": X_type };
}

function parse_prediction(text, prediction) {
    var prediction = prediction.slice(1).concat(1)

    tokenized_text = Array();
    var count = 0;
    var word = "";
    for (var i = 0; i < text.length; i++) {
        word = word + text[i];
        if (prediction[i] > 0.5) {
            tokenized_text[count] = word;
            count = count + 1;
            word = '';
        }
    }
    return tokenized_text;
}
