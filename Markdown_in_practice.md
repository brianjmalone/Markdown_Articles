# Markdown in practice (practice) 

**Bolded text requires two asterisks.**

If want to _italicize_ text, you can enclose the text with _underscores_, or you could use a *single* set of asterisks. 

If you want to show some code, it's likely a good idea to use backticks to `switch to a Monospaced font` that's common in coding examples. More on that below. 

## Dealing with mistakes. 

If it happens, then you need pairs of encosing tildas (~'s) to ~~strikethrough~~ the text. 

### Inserting links 

You put the embedding text in brackets, then follow with the links in parentheses. If you don't believe me, check [Google's Markdown Guide](https://colab.research.google.com/notebooks/markdown_guide.ipynb#scrollTo=Lhfnlq1Surtk). 

The process is similar for an image, but you need a "!" to precede the brackets. Here's an example with this ![symbol](https://www.google.com/images/rss.png). It's from Google. I got it [here](https://www.google.com/images/rss.png). 

Did I just repeat everything *sans* the initial "!" to create that link? 

Yes. 

Did I _italicize_ _sans_ to be pretentious? 

**Also yes**. Was "sans" alone pretentious? `No comment.`

#### Indentation

Yes, the headings are getting smaller as I had more ####'s. 

>Indenting is easy. 

>>The more >'s you use, the more indents you get. 

>>>See? That was three of them. 

##### Code Blocks

So small! Use triple backtics for code, and include the language. Here's an example with Python: 

``` python
print("This is the text I am printing.")
```
Notice the code highlighting? Let's do another one...

``` python
print([num for number in range(10)])
```
The word "python" should appear on the same line as the initial backticks. 

Yeah, you prefer bash. We _all_ do. **FINE**. 

```bash
echo $(pwd)
```
Did I choose that bash example to be annoying? *Kinda*. Anyone who was was annoyed probably didn't like me using "kinda" either. Sticklers. 

### Lists

Yup, **BIGGER** headings again! 

1. One
1. Two
1. Three

Fun fact: I only typed 1's, but Markdown supplied the proper indices. Check the Markdown file. It's not the same as the Preview! Not really a gotcha—more of a "caught 'ya!"

## Bullet Pointed Lists 

Bigger still, since I love these. You might have noticed that pretty lines after the section headings. You get those for free. 

You would think bullet points use •. Nope. It's an asterisk that starts a line followed by a space. That's a tricky one. 

* Look
* at
* this 
* nonsense. 

I *know* right? Wait. I said you can _italicize_ with single asterisks. And I know it's true since that's how I italicized "*know*" in the previous sentence. And this one. 

The trick is keeping the spaces before bullet points. It's ~~stupid~~. I didn't mean that, Markdown. Strike that! (Reminder: ~ is for strikethrough)

## Math is $MONEY$

That is, you have to use the '$' around the equations. But you can write them with syntax familiar from Python. Look how $fancy$ this looks. (Yes, I just wrote "fancy" in math):

$y=x^2$

Not going to lie, I copied **these** from the page I linked above. The following will not be comprehensive. Perhaps not even comprehensible. 

First, this is _everyone's_ favorite equation:

$e^{i\pi} + 1 = 0$

Oddly, infinity is written as "infty". The sum is tricky, and requires \'s and more. You need {}'s for grouping. 

$e^x=\sum_{i=0}^\infty \frac{1}{i!}x^i$

"frac" is for fractions! "\choose" is in there. It's familiar if you've seen nchoosek in MATLAB. 

$\frac{n!}{k!(n-k)!} = {n \choose k}$

This one has a lot of \(c/v/dd)dots and &'s. You've seen the Matrix. Gentle Reader, I copied this one *just for you*. 

$A_{m,n} =
 \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n}
 \end{pmatrix}$

 ## Tables 

 Sorry, folks. Tables require ASCII art, kinda. **_Ugh._** (Yes, you can combine _italics_ and **bolding**, by using _'s for italics. Nice!)

You use |'s to separate the columns. You need one—and only one—on each line to look $pretty$. 

 Column 1  | Another one 
----|------------------
Row 1, Col 1 | Row 1, Col 2 
Row 2, Col 1 | Row 2, Col 2 

Here's a little secret. The underlying ----'s and |'s don't have to line up properly. Get the | counts correct, and you're good. Try messing with it yourself and you will see. Nice try, Google. 

My recommendation if you want to learn Markdown is to repeat this exercise yourself. I wrote it to teach myself. You will internalize it more quickly if you do the same. 

LLMs "speak" Markdown with native fluency. It's how their outputs look so polished. If you interact with them a lot, it's really useful to get familiar with it, since Markdown format is a great way to pass information in and out of them and keep things pretty. 

Github Gists are in Markdown! That's super convenient. 

LLMs also speak HTML. If you use them as editing partners, I find that's an easier path to get nicely formatted documents destined for PDF. Of course, anyone can open HTML in any browser. 

But don't be surprised if they don't, since it seems weird. It's happened to me. 

Hope this helps! 