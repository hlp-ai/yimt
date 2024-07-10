
async function request_ad(){
    const res = await fetch(END_POINT + "/request_ad", {
          method: "POST",
          body: JSON.stringify({platform:"web"}),
          headers: { "Content-Type": "application/json" },
        }
    );
    trans_json = await res.json();
    ad_id = trans_json.ad_id;
    type = trans_json.type;
    content = trans_json.content;
    url = trans_json.url;

     ad_url = document.getElementById("ad_url");
     ad_url.style.visibility='visible';
     ad_url.href = url;

     const base64String = content;
     ad_img_div = document.getElementById("ad_area");
     ad_img_div.style.backgroundImage='url('+'data:image/png;base64,'+content+')';
     ad_img_div.style.backgroundSize = "contain";
     ad_img_div.style.backgroundPosition = "center";
     ad_img_div.style.backgroundRepeat = "no-repeat";
}

async function request_languages(){
    const res = await fetch(END_POINT + "/languages", {method: "GET"});
    ret_json = await res.json();

    console.log("languages: "+ret_json);

    var sel_langs_source = document.getElementById("source");
    var sel_langs_target = document.getElementById("target");

    var opt_src = new Option("自动检测", "auto");  //第一个代表标签内的内容，第二个代表value
	sel_langs_source.options.add(opt_src);

    for(var idx in ret_json){
        console.log(ret_json[idx].code)

        var opt_src = new Option(ret_json[idx].cname, ret_json[idx].code);  //第一个代表标签内的内容，第二个代表value
	    sel_langs_source.options.add(opt_src);
	    var opt_tgt = new Option(ret_json[idx].cname, ret_json[idx].code);  //第一个代表标签内的内容，第二个代表value
	    sel_langs_target.options.add(opt_tgt);
    }
}