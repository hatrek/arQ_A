import flet as ft
import json
def main(page: ft.Page):
    def getsize():
        with open('new_data.json', 'r') as infile:
             my_data = json.load(infile)
        return str(len(my_data['data']))
    def add_clicked(e):
        new_data = {'story': story.value, 'questions': 
    [{"text":ques1.value,"turn_id":1},{"text":ques2.value,"turn_id":2}],
    'answers': [{"text":answ1.value,"turn_id":1},{"text":answ2.value,"turn_id":2}]}
        with open('new_data.json', 'r') as infile:
             my_data = json.load(infile)
    
        my_data['data'] = my_data['data'] + [new_data]

        with open('new_data.json', 'w',encoding='utf-8') as outfile:
            json.dump(my_data, outfile,ensure_ascii=False)
        story.value = ""
        ques1.value = ""
        answ1.value = ""
        ques2.value = ""
        answ2.value = ""
    
        size_dataset.value = 'dataset Size: '+ getsize()
        page.update()
    size_dataset =ft.Text(size=30,color="pink600",)
    size_dataset.value = 'dataset Size: '+ getsize()
    story = ft.TextField(hint_text="story ...", multiline=True,min_lines=3,max_length=512)
    ques1 = ft.TextField(hint_text="Question 1?")
    answ1 = ft.TextField(hint_text="Answering 1")
    ques2 = ft.TextField(hint_text="Question 2?")
    answ2 = ft.TextField(hint_text="Answering 2?")
   

    page.add(story,ques1,answ1,ques2,answ2,size_dataset, ft.FloatingActionButton(icon=ft.icons.ADD, on_click=add_clicked))

ft.app(target=main)