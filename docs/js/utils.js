var CHAR_TYPE = {
    'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    'ฅฉผฟฌหฮ': 'n',
    'ะาำิีืึุู': 'v',  // า ะ ำ ิ ี ึ ื ั ู ุ
    'เแโใไ': 'w',
    '่้๊๋': 't', // วรรณยุกต์ ่ ้ ๊ ๋
    '์ๆฯ.': 's', // ์  ๆ ฯ .
    '0123456789๑๒๓๔๕๖๗๘๙': 'd',
    '"': 'q',
    "‘": 'q',
    "’": 'q',
    "'": 'q',
    ' ': 'p',
    'abcdefghijklmnopqrstuvwxyz': 's_e',
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
};

var CHAR_TYPE_FLATTEN = {};
for (var key in CHAR_TYPE){
    var value = CHAR_TYPE[key];
    for (var i = 0; i < key.length; i++) {
        CHAR_TYPE_FLATTEN[key[i]] = value;
    }
}

CHARS = [
    '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
    ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', '', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'other', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    'z', '}', '~', 'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช',
    'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท',
    'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ',
    'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ฯ', 'ะ', 'ั', 'า',
    'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'ฺ', 'เ', 'แ', 'โ', 'ใ', 'ไ',
    'ๅ', 'ๆ', '็', '่', '้', '๊', '๋', '์', 'ํ', '๐', '๑', '๒', '๓',
    '๔', '๕', '๖', '๗', '๘', '๙', '‘', '’', '\ufeff'
];

CHARS_MAP = {};
for (var i = 0; i < CHARS.length; i++) {
    CHARS_MAP[CHARS[i]] = i;
}

CHAR_TYPES = [
    'b_e', 'c', 'd', 'n', 'o',
    'p', 'q', 's', 's_e', 't',
    'v', 'w'
];

CHAR_TYPES_MAP = {};
for (var i = 0; i < CHAR_TYPES.length; i++) {
    CHAR_TYPES_MAP[CHAR_TYPES[i]] = i;
}

function create_char_dict(text) {
    //Transform input text into list of character feature
    char_dict = []
    for (var i = 0; i < text.length; i++) {
        var char = text[i];
        if (CHARS.indexOf(char) > -1) {
            char_dict.push({'char': char,
                              'type': CHAR_TYPE_FLATTEN.hasOwnProperty(char) ? CHAR_TYPE_FLATTEN[char] : 'o'});
        } else {
            char_dict.push({'char': 'other',
                              'type': CHAR_TYPE_FLATTEN.hasOwnProperty(char) ? CHAR_TYPE_FLATTEN[char] : 'o'});
        }
    }
    return char_dict;
}

function gen(text) {
  var data = []
  for (var i = 0; i < 10; i++) {
    data.push(' ')
  }

  for (var j = 0; j < text.length; j++) {
    data.push(text[j]);
    i++;
  }

  for (var i = 0; i < 10; i++) {
    data.push(' ')
  }

  return data;

}

function gen_input(text) {
  var data = gen(text);
  var x_char = [];
  var x_type = [];
  for (var i = 0; i < data.length; i++) {
    x_char[i] = CHARS_MAP[data[i]];
    var type = CHAR_TYPE_FLATTEN[data[i]];

    if (typeof type == 'undefined') {
      type = 'o';
    }
    x_type[i] = CHAR_TYPES_MAP[type];


  }

  return {"x_type": x_type, "x_char": x_char, "raw": data};
}
