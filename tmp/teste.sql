create table id (id integer primary key autoincrement, e integer, kernelName varchar(30), data varchar(30));
insert into id (e, kernelName, data) values (1, "kernel0", "01");
create table val (id integer primary key autoincrement, e integer, kernelName varchar(30), data varchar(30));
insert into val (e, kernelName, data) values (1, "kernel0", "10");
insert into id (e, kernelName, data) values (1, "kernel0", "02");
insert into val (e, kernelName, data) values (1, "kernel0", "5");
insert into id (e, kernelName, data) values (1, "kernel1", "01");
insert into val (e, kernelName, data) values (1, "kernel1", "90");




