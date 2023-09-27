
"{% load static %}"

var searcher = document.getElementById('Search');

var con = document.getElementsByClassName('trimage');
var c = document.getElementsByClassName('sect');
var co = document.getElementById('trending');
var conn = document.getElementById('Best_S');
var Results_S = document.getElementById('Results_Search');



function rate(id,e) {
    const parent_f =  document.createElement("form");
    parent_f.method='post';

    const newDiv = document.createElement("div");
    parent_f.appendChild(newDiv)

    newDiv.appendChild(document.createTextNode('Rate Movie Id : '+id))
    newDiv.appendChild(document.createElement('Br'));
    var input_m = document.createElement('input');
    var input_rating = document.createElement('input');
    
    input_m.name='movie_id' ;
    input_m.value=id;
    input_m.style.display= 'none';
    newDiv.appendChild(input_m);

    input_rating.name ='ratings' ;
    input_rating.value=0;
    input_rating.style.display= 'none';
    newDiv.appendChild(input_rating);

    var img = document.createElement('img');
    img.id=1;
    var img2 = document.createElement('img');
    img2.id=2;
    var img3 = document.createElement('img');
    img3.id=3;
    var img4 = document.createElement('img');
    img4.id=4;
    var img5 = document.createElement('img');
    img5.id=5;
    
    var done = document.createElement('input');
    done.type='submit';
    done.id='js_done';
    done.value='';

    img.src = "/static/files/empty.png";
    img2.src = "/static/files/empty.png";
    img3.src = "/static/files/empty.png";
    img4.src = "/static/files/empty.png";
    img5.src = "/static/files/empty.png";
    //done.src = "/static/files/tik.png";
    
    newDiv.appendChild(img);
    newDiv.appendChild(img2);
    newDiv.appendChild(img3);
    newDiv.appendChild(img4);
    newDiv.appendChild(img5);
    newDiv.appendChild(done);
    
    done.addEventListener('mousedown',(event)=>{
        parent_f.submit();
    });
    
    parent_f.setAttribute('tabindex', '0');
    parent_f.setAttribute('id','js_');
    parent_f.style.top=e.getBoundingClientRect().top+28+'px';
    parent_f.style.left=e.getBoundingClientRect().left+'px';

    parent_f.addEventListener('blur',(event) => {
        document.body.removeChild(parent_f);
        delete newDiv;
    });
    
    parent_f.addEventListener('focus',(event) => {
    });
    
    clicking = function(e){
        input_rating.value = e.currentTarget.id;
        imgs = newDiv.getElementsByTagName('img');
        for (let i = 0; i < imgs.length; i++) {
           imgs[i].removeEventListener('mouseover',fillers[i]);
           imgs[i].removeEventListener('mouseout',emptiers[i]);
        }
    }
    
    i1 = function(e){
        img.src="/static/files/filled.png";
    };
    y1 = function(e){
        img.src="/static/files/empty.png";
    };
    
    i2 = function(e){
        i1();
        img2.src="/static/files/filled.png";
    };
    y2 = function(e){
        y1();
        img2.src="/static/files/empty.png";
    };
    
    i3 = function(e){
        i2();
        img3.src="/static/files/filled.png";
    };
    y3 = function(e){
        y2();
        img3.src="/static/files/empty.png";
    };
    
    i4 = function(e){
        i3();
        img4.src="/static/files/filled.png";
    };
    y4 = function(e){
        y3();
        img4.src="/static/files/empty.png";
    }
    
    i5 = function(e){
        i4();
        img5.src="/static/files/filled.png";
    };
    y5 = function(e){
        y4();
        img5.src="/static/files/empty.png";
    }
    
    fillers = [i1,i2,i3,i4,i5]
    emptiers = [y1,y2,y3,y4,y5]
    
    imgs = newDiv.getElementsByTagName('img');
    for (let i = 0; i < imgs.length; i++) {
       imgs[i].addEventListener('mouseover',fillers[i]);
       imgs[i].addEventListener('mouseout',emptiers[i]);
       imgs[i].addEventListener('click',clicking);
    }
    
    document.body.appendChild(parent_f);
    parent_f.focus();
}
function recommand(){
    const r_div = document.getElementById('recommandations');
    r_div.style.display ="block";
}
function searchBytitle(){
co.style.display='none';
con[0].style.display='none';
con[1].style.display='none';
c[0].style.display='none';
c[1].style.display='none';
conn.style.display='none';
Results_S.style.display='block';
}
function searchByCategory(){
    document.getElementById('my_form').submit();
    co.style.display='none';
    con[0].style.display='none';
    con[1].style.display='none';
    c[0].style.display='none';
    c[1].style.display='none';
    conn.style.display='none';
    Results_S.style.display='block';
}
