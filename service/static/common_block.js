

function display_setting_func()
{
    document.getElementById("settings_block").style.visibility='visible';
    document.getElementById("settings_mask").style.display='block';
    document.getElementById("server_url").value=END_POINT;
    document.getElementById("api_key").value=API_KEY;
}

function reset_setting_func()
{
    END_POINT = document.getElementById("server_url").value;
    window.localStorage.setItem("server_url", END_POINT);

    document.getElementById("url_setting_block").style.visibility='hidden';
    document.getElementById("mask").style.display='none';

    API_KEY = document.getElementById("api_key").value;
    window.localStorage.setItem("key",API_KEY);
}

function close_setting_func()
{
    document.getElementById("settings_block").style.visibility='hidden';
    document.getElementById("settings_mask").style.display='none'
}

async function request_ad(){
    const res = await fetch("/request_ad", {
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
     //ad_url.href = url;
     ad_url.href = "/click_ad?ad_id=" + ad_id + "&platform=web" + "&url=" + url;

     const base64String = content;
     ad_img_div = document.getElementById("ad_area");
     ad_img_div.style.backgroundImage='url('+'data:image/png;base64,'+content+')';
     ad_img_div.style.backgroundSize = "contain";
     ad_img_div.style.backgroundPosition = "center";
     ad_img_div.style.backgroundRepeat = "no-repeat";
}

async function request_languages(){
    const res = await fetch("/languages", {method: "GET"});
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

 function isUrl(s) {
   var regexp = /(http|https):\/\/(\w+:{0,1}\w*@)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%@!\-\/]))?/
   return regexp.test(s);
}

