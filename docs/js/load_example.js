const example_list = [
    'เราจะทำตามสัญญา ขอเวลาอีกไม่นาน\
    แล้วแผ่นดินที่งดงามจะคืนกลับมา\
    เราจะทำอย่างซื่อตรง ขอแค่เธอจงไว้ใจและศรัทธา\
    แผ่นดินจะดีในไม่ช้า ขอคืนความสุขให้เธอ ประชาชน',

    "'บิ๊กตู่' แจงเล็งออกกฎหมายห้ามมีกิ๊กแค่พูดเล่น บอกเป็นคนตลก \
    ซัดสื่อสนใจแต่เรื่องขี้ไก่ 'วิษณุ'ชี้ 'ไม่มีสาระเลย' เลขาฯกฤษฎีกา\
    เผยร่างกฎหมาย 4 ชั่วโคตรแค่ตีกรอบคนมีคู่ไม่จดทะเบียนสมรส",

    "คิดว่ามันเป็นเพียงแค่ความฝัน ไม่เคยหวัง ไม่เคยคิดจริงจังอะไร \
    แค่แอบหลงรักเธอเล่นๆ ตามลำพังข้างเดียวในหัวใจ \
    แต่เมื่อมารู้สึกนึกอีกที เมื่อรู้ตัวอีกที ฉันก็รักจนลึกข้างใน",

    "สีงามอร่ามหรูชมพู-ฟ้า\
    งามยั่วยวนชวนประชาให้เห็นเด่น\
    งามจริงยิ่งเห็นเป็นโชคชัย",

    "เพียงแค่ฉันเองก็ไม่รู้ว่าในทุกๆวันที่มีแต่เธอ \
    ยังคงละเมอและเพ้อทุกครั้งที่มองขึ้นไป \
    บนท้องฟ้าหยดน้ำค้างตกกระทบในตาฉัน \
    แล้วในวันนี้พรุ่งนี้จะเป็นยังไงจะเดินจะกิน \
    จะนอนจนวันนึงให้วันเวลาเดินไปเท่าไร \
    แต่สุดท้ายแดดยามเช้าก็ไม่สดใส"
]

// helper function
function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min)) + min; //The maximum is exclusive and the minimum is inclusive
}

function onClickLoadExample() {
    // set val of textarea by random from example list then set height to 0
    // and resize to scroll height (*this should be fixed more elegantly)
    // finally focus on textarea to show cursor and trigger keyup for prediction
    $('#message').val(function (i, text) {
        return example_list[getRandomInt(0, example_list.length)];
    }).height(0).height(this.scrollHeight).focus().trigger('keyup');
}
