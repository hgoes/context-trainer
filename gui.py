import gtk
from annpkg.model import AnnPkg
import gobject
import numpy as np
import threading
import rule
import json
import math
import Training

gobject.threads_init()

class TrainingProgress(threading.Thread):
    def __init__(self,mp,training_data,progress,done):
        threading.Thread.__init__(self)
        self.mp = mp
        self.training_data = training_data
        self.progress = progress
        self.done = done
    def cb(self,pc):
        self.progress.set_fraction(pc)
    def run(self):
        state = Training.TrainingState()
        self.cls_id = 0
        idx = 1
        for key,val in self.mp.iteritems():
            classifier_state = Training.ClassifierState(key)
            for cls in val:
                tr_dat = []
                ch_dat = []
                for i,dat in enumerate(self.training_data[cls]):
                    if i % 2 == 0:
                        tr_dat.append(dat)
                    else:
                        ch_dat.append(dat)
                classifier_state.add_class_state(Training.ClassState(cls,idx,np.array(tr_dat),np.array(ch_dat)))
                idx += 1

            # Build complementary class:
            if len(self.mp) > 1:
                tr_dat = []
                ch_dat = []
                for key2,val2 in self.mp.iteritems():
                    if key != key2:
                        for cls in val2:
                            for i,dat in enumerate(self.training_data[cls]):
                                if i % 20 == 0:
                                    tr_dat.append(dat)
                                elif i % 20 == 1:
                                    ch_dat.append(dat)
                classifier_state.add_class_state(Training.ClassState(None,0,np.array(tr_dat),np.array(ch_dat)))
            
            #fis = buildFIS(tr_lst,iterations=3,cb=self.cb,check_data=ch_lst)
            #for cls in val:
            #    for d in self.training_data[cls]:
            #        d2 = np.empty(d.shape[0]+1)
            #        d2[0:d.shape[0]] = d
            #        d2[-1] = 0.0
            #        print cls,fis.evaluate(d2)
            #result.append(rule.Classifier(fis,memb,key))
            #self.cls_id += 1
            state.add_classifier_state(classifier_state)
        self.done(state.buildFIS(iterations=1,cb=self.cb))
        

class ContextTrainer(gtk.Assistant):
    TARGETS = [('MY_TREE_MODEL_ROW', gtk.TARGET_SAME_WIDGET, 0),
               ('text/plain', 0, 1),
               ('TEXT', 0, 2),
               ('STRING', 0, 3)]
    def __init__(self):
        gtk.Assistant.__init__(self)
        self.create_page1()
        self.create_page2()
        self.create_page3()
        self.create_page4()
        self.connect('prepare',self.prepare_page)
        self.connect('apply',self.finish_assistant)
        self.connect('destroy', lambda x: gtk.main_quit())
        w = 600
        phi = (1 + math.sqrt(5)) / 2
        self.set_default_size(w,int(w / phi))
    def create_page1(self):
        p1 = gtk.VBox()
        lbl_select = gtk.Label("Please choose a training data set. Only the annotated data from the data set will be used.")
        lbl_select.set_alignment(0.0,0.0)
        lbl_select.set_padding(5,0)
        lbl_select.set_line_wrap(True)
        lbl_select.set_size_request(500,-1)
        box_data = gtk.HBox()
        lbl_training = gtk.Label()
        lbl_training.set_markup("<b>Training data set:</b>")
        training_data_selector = gtk.FileChooserButton("Select training data set")
        tar_filter = gtk.FileFilter()
        tar_filter.set_name("Annotated data set")
        tar_filter.add_pattern("*.tar")
        training_data_selector.add_filter(tar_filter)
        training_data_selector.connect('selection_changed',self.chooser_changed)
        box_data.pack_start(lbl_training,expand=False,padding=5)
        box_data.pack_start(training_data_selector,expand=True,padding=5)
        p1.pack_start(lbl_select,expand=True,padding=5)
        p1.pack_start(box_data,expand=False,padding=5)
        self.append_page(p1)
        self.set_page_title(p1,"Data selection")
        self.chooser = training_data_selector
        self.page1 = p1
    def chooser_changed(self,fc):
        self.training_data = None
        self.set_page_complete(self.page1,True)
    def load_ann_pkg(self):
        ann_pkg = AnnPkg.load(self.chooser.get_filename())
        mp = {}

        for dat in ann_pkg.movement_data():
            if dat[0] in mp:
                mp[dat[0]].append(np.array(dat[1:]))
            else:
                mp[dat[0]] = [np.array(dat[1:])]
        self.training_data = mp
    def prepare_page(self,ass,page):
        if page == self.page1:
            pass
        elif page == self.page2:
            if self.training_data is None:
                self.load_ann_pkg()
                self.model = ClassifierAssoc(self.training_data.keys())
                self.tv.set_model(self.model)
            self.set_page_complete(self.page2,True)
        elif page == self.page3:
            thr = TrainingProgress(self.model.get_mapping(),self.training_data,self.progress,self.finish_3)
            thr.start()
            
            
    def create_page2(self):
        box = gtk.VBox()
        tv = gtk.TreeView()
        cr = gtk.CellRendererText()
        col = gtk.TreeViewColumn("Name",cr,text=0)
        tv.append_column(col)
        but = gtk.Button(stock=gtk.STOCK_ADD)
        but.connect('clicked',self.add_classifier)
        box.pack_start(tv,expand=True)
        box.pack_start(but,expand=False)
        tv.enable_model_drag_source(gtk.gdk.BUTTON1_MASK,self.TARGETS,gtk.gdk.ACTION_DEFAULT|gtk.gdk.ACTION_MOVE)
        tv.enable_model_drag_dest(self.TARGETS,gtk.gdk.ACTION_DEFAULT)
        #tv.drag_source_set(gtk.gdk.BUTTON1_MASK,self.TARGETS,gtk.gdk.ACTION_DEFAULT|gtk.gdk.ACTION_MOVE)
        tv.connect("drag_data_get",self.drag_data_get_data)
        tv.connect("drag_data_received",self.drag_data_received_data)
        self.append_page(box)
        self.set_page_title(box,"Classifier preperation")
        self.tv = tv
        self.page2 = box
    def drag_data_get_data(self, treeview, context, selection, target_id,etime):
        select = treeview.get_selection()
        model,iter = select.get_selected()
        data = model.get_value(iter,1)
        if data:
            if data[0] == 'class':
                selection.set(selection.target,8,data[1])
    def drag_data_received_data(self, treeview, context, x, y, selection,info,etime):
        model = treeview.get_model()
        data = selection.data
        if data is None:
            context.finish(False,False,etime)
            return
        drop_info = treeview.get_dest_row_at_pos(x, y)
        if drop_info:
            path, pos = drop_info
            iter = model.get_iter(path)
            data2 = model.get_value(iter,1)
            if data2 is None:
                model.add_unassigned(data)
                if context.action == gtk.gdk.ACTION_MOVE:
                    context.finish(True,True,etime)
            else:
                if data2[0] == 'classifier':
                    if pos == gtk.TREE_VIEW_DROP_INTO_OR_BEFORE or pos == gtk.TREE_VIEW_DROP_INTO_OR_AFTER:
                        model.append(iter,(data,('class',data)))
                        if context.action == gtk.gdk.ACTION_MOVE:
                            context.finish(True,True,etime)
                elif data2[0] == 'class':
                    if pos == gtk.TREE_VIEW_DROP_INTO_OR_BEFORE or pos == gtk.TREE_VIEW_DROP_BEFORE:
                        model.insert_before(None,iter,(data,('class',data)))
                    else:
                        model.insert_after(None,iter,(data,('class',data)))
                    if context.action == gtk.gdk.ACTION_MOVE:
                        context.finish(True,True,etime)
                    
    def add_classifier(self,but):
        dialog = gtk.MessageDialog(None,
                                   gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
                                   gtk.MESSAGE_QUESTION,
                                   gtk.BUTTONS_OK, None)
        dialog.set_markup("Please enter the <b>name</b> of the classifier")
        entry = gtk.Entry()
        entry.connect("activate", lambda wid: dialog.response(gtk.RESPONSE_OK))
        dialog.vbox.pack_end(entry,expand=True,fill=True)
        dialog.show_all()
        dialog.run()
        self.model.add_classifier(entry.get_text())
        dialog.destroy()
    def finish_3(self,fis):
        self.set_page_complete(self.page3,True)
        self.fis = fis
    def create_page3(self):
        box = gtk.VBox()
        lbl = gtk.Label("Please wait while generating classifiers...")
        lbl.set_alignment(0.0,0.0)
        lbl.set_padding(5,0)
        lbl.set_line_wrap(True)
        lbl.set_size_request(500,-1)
        self.progress = gtk.ProgressBar()
        box.pack_start(lbl,expand=True,padding=5)
        box.pack_start(self.progress,expand=False,padding=5)
        self.append_page(box)
        self.set_page_title(box,"Creating classifiers")
        self.set_page_type(box,gtk.ASSISTANT_PAGE_PROGRESS)
        self.page3 = box
    def create_page4(self):
        p4 = gtk.VBox()
        lbl_select = gtk.Label("Please provide the file to which to write the classifier. The classifier is written in JSON format.")
        lbl_select.set_alignment(0.0,0.0)
        lbl_select.set_padding(5,0)
        lbl_select.set_line_wrap(True)
        lbl_select.set_size_request(500,-1)
        box_data = gtk.HBox()
        lbl_training = gtk.Label()
        lbl_training.set_markup("<b>Classifier file:</b>")
        self.classifier_selector = gtk.FileChooserButton(title="Save classifier file")
        #classifier_selector = gtk.FileChooserWidget(action=gtk.FILE_CHOOSER_ACTION_SAVE)
        self.classifier_selector.set_action(gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER)
        json_filter = gtk.FileFilter()
        json_filter.set_name("Classifier file")
        json_filter.add_pattern("*.json")
        self.classifier_selector.add_filter(json_filter)
        self.entry = gtk.Entry()
        self.entry.connect('changed',self.saver_changed)
        box_data.pack_start(lbl_training,expand=False,padding=5)
        box_data.pack_start(self.entry,expand=True,padding=5)
        box_data.pack_start(self.classifier_selector,expand=True,padding=5)
        p4.pack_start(lbl_select,expand=True,padding=5)
        p4.pack_start(box_data,expand=False,padding=5)
        self.append_page(p4)
        self.set_page_title(p4,"Saving")
        self.set_page_type(p4,gtk.ASSISTANT_PAGE_CONFIRM)
        self.page4 = p4
    def saver_changed(self,fc):
        self.set_page_complete(self.page4,fc.get_text() != "")
    def finish_assistant(self,ass):
        with open(self.classifier_selector.get_filename()+'/'+self.entry.get_text(),'w') as h:
            json.dump(self.fis.to_json(),h,indent=True)
        gtk.main_quit()

class ClassifierAssoc(gtk.TreeStore):
    def __init__(self,classes):
        gtk.TreeStore.__init__(self,str,object)
        self.unassigned = self.insert(None,0,("Unassigned",None))
        last = None
        for cls in classes:
            last = self.insert_after(self.unassigned,last,(cls,('class',cls)))
    def add_classifier(self,name):
        self.append(None,("Classifier "+name,('classifier',name)))
    def add_unassigned(self,name):
        self.append(self.unassigned,(name,('class',name)))
    def get_mapping(self):
        cur = self.get_iter_root()
        mp = {}
        while cur:
            data = self.get_value(cur,1)
            if data:
                if data[0] == 'classifier':
                    cls = []
                    child = self.iter_children(cur)
                    while child:
                        cls.append(self.get_value(child,1)[1])
                        child = self.iter_next(child)
                    mp[data[1]] = cls
            cur = self.iter_next(cur)
        return mp

if __name__ == "__main__":
    gtk.gdk.threads_init()
    test = ContextTrainer()
    test.show_all()

    gtk.main()
